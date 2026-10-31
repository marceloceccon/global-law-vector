"""`glv tui`: choose corpora with live cost totals, run the indexer, search the index.

Everything slow (scanning 4.5 million inodes, talking to the database or the embeddings server)
runs in worker threads; the screens only render state.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Log, ProgressBar, Static, Tree

from ..config import Config
from ..discover import Corpus, CorpusStats, StatsCache, discover, stats_for
from ..discover import Tree as LawTree
from ..index import Progress
from ..selection import Selection, estimate, human_bytes, human_duration

log = logging.getLogger(__name__)


@dataclass
class State:
    cfg: Config
    tree: LawTree
    selection: Selection
    stats: dict[str, CorpusStats]
    status: dict[str, str]  # corpus path -> "indexed 2026-10-30" / "stale" / "partial"


# --- select screen -----------------------------------------------------------------------------


class CorpusTree(Tree):
    """A Tree whose space key toggles selection instead of expanding the node."""

    BINDINGS = [
        Binding("space", "toggle_selection", "toggle"),
        Binding("enter", "toggle_node", "expand/collapse", show=False),
    ]

    def action_toggle_selection(self) -> None:
        self.screen.action_toggle()


class SelectScreen(Screen):
    BINDINGS = [
        Binding("a", "select_all", "all"),
        Binding("n", "select_none", "none"),
        Binding("slash", "focus_filter", "filter"),
        Binding("s", "save", "save selection"),
        Binding("r", "run", "run"),
        Binding("f", "search", "search"),
        Binding("escape", "focus_tree", "tree", show=False),
        Binding("q", "quit", "quit"),
    ]

    class StatsReady(Message):
        def __init__(self, stats: dict[str, CorpusStats]) -> None:
            super().__init__()
            self.stats = stats

    class StatusReady(Message):
        def __init__(self, status: dict[str, str]) -> None:
            super().__init__()
            self.status = status

    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state
        self._filter = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="filter countries and corpora (press / to focus)", id="filter")
        yield CorpusTree("global-law", id="tree")
        yield Static("", id="totals")
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = f"model: {self.state.cfg.model.split('/')[-1]} ({self.state.cfg.dims})"
        self.rebuild()
        self.query_one("#tree", Tree).focus()
        self.load_stats()
        self.load_status()

    # --- background loads ---

    @work(thread=True, exclusive=True, group="stats")
    def load_stats(self) -> None:
        st = self.state
        cache = StatsCache(st.cfg.cache_dir, st.tree.root)
        pending = [c for c in st.tree.corpora if c.path not in st.stats]
        batch: list[Corpus] = []
        for c in pending:
            batch.append(c)
            if len(batch) >= 25:
                self.post_message(self.StatsReady(stats_for(st.tree, batch, cache)))
                batch = []
        if batch:
            self.post_message(self.StatsReady(stats_for(st.tree, batch, cache)))

    @work(thread=True, exclusive=True, group="status")
    def load_status(self) -> None:
        try:
            from ..store import Store

            with Store(self.state.cfg.database_url, self.state.cfg.schema,
                       self.state.cfg.vector_type, self.state.cfg.dims) as store:
                store.migrate()
                rows = store.corpus_rows()
        except Exception as exc:  # noqa: BLE001 - the DB is optional for selecting
            log.info("database not reachable: %s", exc)
            return
        by = self.state.tree.by_path()
        status: dict[str, str] = {}
        for path, r in rows.items():
            c = by.get(path)
            if r.completed_at is None:
                status[path] = "◐ partial"
            elif c is not None and r.commit_sha != c.commit_sha:
                status[path] = "◐ stale"
            else:
                status[path] = "● indexed " + r.completed_at.strftime("%Y-%m-%d")
        self.post_message(self.StatusReady(status))

    @on(StatsReady)
    def _stats_ready(self, msg: StatsReady) -> None:
        self.state.stats.update(msg.stats)
        self.refresh_labels()

    @on(StatusReady)
    def _status_ready(self, msg: StatusReady) -> None:
        self.state.status = msg.status
        self.refresh_labels()

    # --- tree ---

    def rebuild(self) -> None:
        tree = self.query_one("#tree", Tree)
        tree.clear()
        tree.show_root = False
        f = self._filter.lower()
        for country, corpora in self.state.tree.countries().items():
            matching = [c for c in corpora if not f or f in c.path.lower()]
            if not matching:
                continue
            node = tree.root.add("", data=("country", country), expand=bool(f))
            for c in matching:
                node.add_leaf("", data=("corpus", c.path))
        self.refresh_labels()

    def _corpus_label(self, c: Corpus) -> str:
        sel = self.state.selection.is_selected(c)
        s = self.state.stats.get(c.path)
        count = f"{s.file_count:>9,} files {human_bytes(s.byte_count):>9}" if s else "scanning…"
        status = self.state.status.get(c.path, "○ not indexed")
        lang = f" [{c.language}]" if c.language else ""
        return f"[{'x' if sel else ' '}] {c.short:<36}{lang:<6} {count}   {status}"

    def _country_label(self, country: str, corpora: list[Corpus]) -> str:
        n_sel = sum(self.state.selection.is_selected(c) for c in corpora)
        mark = "x" if n_sel == len(corpora) else ("-" if n_sel else " ")
        stats = [self.state.stats[c.path] for c in corpora if c.path in self.state.stats]
        files = sum(s.file_count for s in stats)
        size = sum(s.byte_count for s in stats)
        est = estimate({s.path: s for s in stats}, self.state.cfg.chunks_per_second)
        tail = (f"{files:>9,} files {human_bytes(size):>9}   ~{est.human_time}"
                if len(stats) == len(corpora) else "scanning…")
        return f"[{mark}] {country:<36} {len(corpora):>3} corpora {tail}"

    def refresh_labels(self) -> None:
        tree = self.query_one("#tree", Tree)
        by = self.state.tree.by_path()
        countries = self.state.tree.countries()
        for node in tree.root.children:
            kind, name = node.data
            node.set_label(Text(self._country_label(name, countries[name])))
            for leaf in node.children:
                leaf.set_label(Text(self._corpus_label(by[leaf.data[1]])))
        self.refresh_totals()

    def refresh_totals(self) -> None:
        st = self.state
        try:
            chosen = st.selection.resolve(st.tree)
        except KeyError as exc:
            self.query_one("#totals", Static).update(f"selection error: {exc.args[0]}")
            return
        stats = {c.path: st.stats[c.path] for c in chosen if c.path in st.stats}
        est = estimate(stats, st.cfg.chunks_per_second, st.selection.max_files_per_corpus)
        missing = len(chosen) - len(stats)
        text = (f"Selected: {len(chosen)} corpora · {est.files:,} files · {est.human_bytes} · "
                f"~{est.chunks:,} chunks · est. {est.human_time} at "
                f"{st.cfg.chunks_per_second:.0f} chunks/s")
        if missing:
            text += f"   ({missing} still scanning)"
        self.query_one("#totals", Static).update(text)

    # --- actions ---

    def action_toggle(self) -> None:
        tree = self.query_one("#tree", Tree)
        node = tree.cursor_node
        if node is None or node.data is None:
            return
        kind, name = node.data
        sel = self.state.selection
        if kind == "corpus":
            c = self.state.tree.by_path()[name]
            if sel.is_selected(c):
                sel.exclude(corpora=[name])
            else:
                sel.include(corpora=[name])
        else:
            corpora = self.state.tree.countries()[name]
            if all(sel.is_selected(c) for c in corpora):
                sel.exclude(countries=[name])
                sel.exclude(corpora=[c.path for c in corpora])
            else:
                sel.exclude_corpora = sorted(set(sel.exclude_corpora) - {c.path for c in corpora})
                sel.include(countries=[name])
        self.refresh_labels()

    def action_select_all(self) -> None:
        self.state.selection.clear()
        self.state.selection.all = True
        self.refresh_labels()

    def action_select_none(self) -> None:
        self.state.selection.clear()
        self.refresh_labels()

    def action_focus_filter(self) -> None:
        self.query_one("#filter", Input).focus()

    def action_focus_tree(self) -> None:
        self.query_one("#tree", Tree).focus()

    @on(Input.Changed, "#filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self._filter = event.value.strip()
        self.rebuild()

    @on(Input.Submitted, "#filter")
    def _filter_submitted(self) -> None:
        self.action_focus_tree()

    def action_save(self) -> None:
        self.state.selection.save(self.state.cfg.selection_file)
        self.notify(f"saved {self.state.cfg.selection_file}")

    def action_run(self) -> None:
        if self.state.selection.is_empty():
            self.notify("nothing selected", severity="warning")
            return
        self.action_save()
        self.app.push_screen(RunScreen(self.state))

    def action_search(self) -> None:
        self.app.push_screen(SearchScreen(self.state))


# --- run screen --------------------------------------------------------------------------------


class RunScreen(Screen):
    BINDINGS = [
        Binding("escape", "stop", "stop (safe: resumes next time)"),
        Binding("q", "back", "back", show=False),
    ]

    class Tick(Message):
        def __init__(self, p: Progress) -> None:
            super().__init__()
            self.p = p

    class Done(Message):
        def __init__(self, text: str, error: bool = False) -> None:
            super().__init__()
            self.text = text
            self.error = error

    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state
        self._stop = threading.Event()
        self._finished = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("preparing…", id="line")
            yield ProgressBar(total=100, id="bar")
            yield Log(id="log", highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = "indexing"
        self.run_index()

    @work(thread=True, exclusive=True, group="index")
    def run_index(self) -> None:
        from ..embed import TEIClient, check_model
        from ..index import Indexer, plan_corpus
        from ..store import Store
        from ..tokenizer import load_tokenizer

        st, cfg = self.state, self.state.cfg
        logw = self.query_one("#log", Log)

        def say(text: str) -> None:
            self.app.call_from_thread(logw.write_line, text)

        try:
            corpora = st.selection.resolve(st.tree)
            client = TEIClient(cfg.tei_url, timeout=cfg.tei_timeout)
            info = client.info()
            check_model(info, cfg.model, cfg.dims, client)
            say(f"embeddings: {info.model_id} at {cfg.tei_url}")
            tok = load_tokenizer(cfg.model, cfg.tokenizer_path, cache_dir=cfg.cache_dir)
            say(f"tokenizer: {tok.name}")
            with Store(cfg.database_url, cfg.schema, cfg.vector_type, cfg.dims) as store:
                store.migrate()
                store.ensure_meta(cfg.model, tok.name, cfg.chunk_tokens)
                plans = []
                for c in corpora:
                    if self._stop.is_set():
                        break
                    pl = plan_corpus(st.tree, c, store, st.selection.max_files_per_corpus)
                    plans.append(pl)
                    say(f"{c.path}: {len(pl.to_index):,} to index, {pl.unchanged:,} unchanged, "
                        f"{len(pl.to_delete):,} to delete")
                indexer = Indexer(cfg, st.tree, store, client, tok,
                                  progress=lambda p: self.post_message(self.Tick(p)),
                                  stop=self._stop.is_set)
                p = indexer.run(plans)
                if not self._stop.is_set() and not store.has_hnsw() and p.chunks_done:
                    say("building HNSW index…")
                    store.build_hnsw()
            text = (f"{'stopped' if self._stop.is_set() else 'done'}: {p.files_done:,} files, "
                    f"{p.chunks_done:,} chunks in {human_duration(p.elapsed)} "
                    f"({p.chunks_per_second:.1f} chunks/s)")
            self.post_message(self.Done(text))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            log.exception("indexing failed")
            self.post_message(self.Done(f"failed: {exc}", error=True))

    @on(Tick)
    def _tick(self, msg: Tick) -> None:
        p = msg.p
        bar = self.query_one("#bar", ProgressBar)
        bar.update(total=max(p.files_total, 1), progress=p.files_done)
        rate = p.chunks_per_second
        eta = human_duration((p.files_total - p.files_done) * 1.05 / rate) if rate else "…"
        self.query_one("#line", Static).update(
            f"{p.corpus}   files {p.files_done:,}/{p.files_total:,}   chunks {p.chunks_done:,}   "
            f"{rate:.1f} chunks/s   eta {eta}")

    @on(Done)
    def _done(self, msg: Done) -> None:
        self._finished = True
        self.query_one("#line", Static).update(Text(msg.text))
        self.query_one("#log", Log).write_line(msg.text)
        self.notify(msg.text, severity="error" if msg.error else "information", timeout=10)

    def action_stop(self) -> None:
        if self._finished:
            self.app.pop_screen()
            return
        self._stop.set()
        self.query_one("#log", Log).write_line("stopping after the current group…")

    def action_back(self) -> None:
        if self._finished:
            self.app.pop_screen()


# --- search screen -----------------------------------------------------------------------------


class SearchScreen(Screen):
    BINDINGS = [Binding("escape", "back", "back")]

    class Results(Message):
        def __init__(self, hits: list, error: str | None = None) -> None:
            super().__init__()
            self.hits = hits
            self.error = error

    def __init__(self, state: State) -> None:
        super().__init__()
        self.state = state
        self._hits: list = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="bar"):
            yield Input(placeholder="ask in any language…", id="query")
            yield Input(placeholder="country filter (optional)", id="country")
        yield DataTable(id="results", cursor_type="row")
        yield Static("", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self.app.sub_title = "search"
        t = self.query_one("#results", DataTable)
        t.add_columns("score", "country", "path", "title")
        self.query_one("#query", Input).focus()

    @on(Input.Submitted)
    def _submitted(self) -> None:
        q = self.query_one("#query", Input).value.strip()
        country = self.query_one("#country", Input).value.strip()
        if q:
            self.do_search(q, [country] if country else [])

    @work(thread=True, exclusive=True, group="search")
    def do_search(self, q: str, countries: list[str]) -> None:
        from ..embed import TEIClient
        from ..search import search
        from ..store import Store

        cfg = self.state.cfg
        try:
            with Store(cfg.database_url, cfg.schema, cfg.vector_type, cfg.dims) as store:
                hits = search(store, TEIClient(cfg.tei_url, timeout=cfg.tei_timeout), q,
                              query_prefix=cfg.query_prefix,
                              top=20, countries=countries)
            self.post_message(self.Results(hits))
        except Exception as exc:  # noqa: BLE001
            self.post_message(self.Results([], error=str(exc)))

    @on(Results)
    def _results(self, msg: Results) -> None:
        t = self.query_one("#results", DataTable)
        t.clear()
        self._hits = msg.hits
        if msg.error:
            self.query_one("#detail", Static).update(Text(f"search failed: {msg.error}"))
            return
        for h in msg.hits:
            t.add_row(f"{h.score:.3f}", h.country, h.path, Text((h.title or "")[:60]))
        if msg.hits:
            t.focus()
            self._show(0)
        else:
            self.query_one("#detail", Static).update("no results")

    @on(DataTable.RowHighlighted)
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        if event.cursor_row is not None:
            self._show(event.cursor_row)

    def _show(self, i: int) -> None:
        if not (0 <= i < len(self._hits)):
            return
        h = self._hits[i]
        head = f"{h.path}  (corpus commit {h.commit_sha or '?'}, chunk {h.ordinal})"
        if h.citation:
            head += f"  [{h.citation}]"
        if h.heading_path:
            head += f"\n{h.heading_path}"
        self.query_one("#detail", Static).update(Text(f"{head}\n\n{h.text}"))

    def action_back(self) -> None:
        self.app.pop_screen()


# --- app ---------------------------------------------------------------------------------------


class GlvApp(App):
    TITLE = "global-law-vector"
    CSS = """
    #tree { height: 1fr; }
    #totals { height: 1; padding: 0 1; background: $primary-background; }
    #filter { margin: 0 1; }
    #line { height: 1; padding: 0 1; }
    #bar { height: 3; }
    #log { height: 1fr; }
    #results { height: 1fr; }
    #detail { height: 12; padding: 0 1; border-top: solid $primary; overflow-y: auto; }
    #query { width: 3fr; }
    #country { width: 1fr; }
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    def on_mount(self) -> None:
        from ..index import resolve_root

        tree = discover(resolve_root(self.cfg))
        state = State(self.cfg, tree, Selection.load(self.cfg.selection_file), {}, {})
        self.push_screen(SelectScreen(state))


def run_tui(cfg: Config) -> int:
    GlvApp(cfg).run()
    return 0
