"""`glv`: select, index, status, search, tui.

Standard-library argparse; the TUI is imported only when asked for, so the package works
without Textual installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import Config
from .discover import StatsCache, Tree, discover, stats_for
from .index import Indexer, Progress, dry_run_counts, plan_corpus, resolve_root
from .selection import Selection, estimate, human_bytes, human_duration

log = logging.getLogger("glv")


# --- helpers -----------------------------------------------------------------------------------


def _tree(cfg: Config) -> Tree:
    return discover(resolve_root(cfg))


def _selection_from_args(
    cfg: Config, args: argparse.Namespace, base: Selection | None
) -> Selection:
    """The saved selection, overridden by one-off flags when any are given."""
    sel = base if base is not None else Selection.load(cfg.selection_file)
    flags = any(getattr(args, k, None) for k in ("country", "corpus", "all"))
    if flags:
        sel = Selection(all=bool(getattr(args, "all", False)))
        sel.include(countries=args.country or [], corpora=args.corpus or [])
    if getattr(args, "exclude", None):
        countries = [x for x in args.exclude if "/" not in x]
        corpora = [x for x in args.exclude if "/" in x]
        sel.exclude(countries=countries, corpora=corpora)
    if getattr(args, "max_files", None) is not None:
        sel.max_files_per_corpus = args.max_files
    return sel


def _print_table(rows: list[Sequence[str]], header: Sequence[str]) -> None:
    widths = [len(h) for h in header]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


def _show_selection(cfg: Config, tree: Tree, sel: Selection, with_stats: bool = True) -> int:
    try:
        corpora = sel.resolve(tree)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    if not corpora:
        print("selection is empty: add countries or corpora with `glv select --country …`")
        return 0
    if not with_stats:
        for c in corpora:
            print(c.path)
        return 0
    cache = StatsCache(cfg.cache_dir, tree.root)
    stats = stats_for(tree, corpora, cache, progress=lambda c, i, n: print(
        f"\rscanning {i + 1}/{n} {c.path:<60}", end="", file=sys.stderr))
    print("\r" + " " * 90 + "\r", end="", file=sys.stderr)
    rows = []
    for c in corpora:
        s = stats[c.path]
        rows.append((c.path, c.language or "", f"{s.file_count:,}", human_bytes(s.byte_count),
                     (c.commit_sha or "")[:10]))
    _print_table(rows, ("corpus", "lang", "files", "size", "commit"))
    est = estimate(stats, cfg.chunks_per_second, sel.max_files_per_corpus)
    print()
    print(f"{est.corpora} corpora · {est.files:,} files · {est.human_bytes} · "
          f"~{est.chunks:,} chunks · est. {est.human_time} at {cfg.chunks_per_second:.0f} chunks/s")
    return 0


# --- commands ----------------------------------------------------------------------------------


def cmd_select(cfg: Config, args: argparse.Namespace) -> int:
    tree = _tree(cfg)
    sel = Selection.load(cfg.selection_file)
    if args.clear:
        sel.clear()
    if args.all:
        sel.all = True
    if args.country or args.corpus:
        sel.include(countries=args.country or [], corpora=args.corpus or [])
    if args.exclude:
        sel.exclude(countries=[x for x in args.exclude if "/" not in x],
                    corpora=[x for x in args.exclude if "/" in x])
    if args.max_files is not None:
        sel.max_files_per_corpus = args.max_files
    try:
        sel.resolve(tree)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    changed = any([args.clear, args.all, args.country, args.corpus, args.exclude,
                   args.max_files is not None])
    if changed:
        sel.save(cfg.selection_file)
        print(f"saved {cfg.selection_file}")
    if args.show or not changed:
        return _show_selection(cfg, tree, sel, with_stats=not args.paths)
    return 0


def cmd_list(cfg: Config, args: argparse.Namespace) -> int:
    tree = _tree(cfg)
    sel = Selection.load(cfg.selection_file)
    rows = []
    for c in tree.corpora:
        if args.country and c.country not in args.country:
            continue
        rows.append(("x" if sel.is_selected(c) else " ", c.path, c.language or "",
                     c.label or "", (c.commit_sha or "")[:10]))
    _print_table(rows, ("sel", "corpus", "lang", "label", "commit"))
    print(f"\n{len(rows)} corpora in {len({r[1].split('/')[0] for r in rows})} countries")
    return 0


def _open_store(cfg: Config):
    from .store import Store

    return Store(cfg.database_url, cfg.schema, cfg.vector_type, cfg.dims)


def cmd_index(cfg: Config, args: argparse.Namespace) -> int:
    from .embed import TEIClient, check_model
    from .tokenizer import load_tokenizer

    tree = _tree(cfg)
    sel = _selection_from_args(cfg, args, None)
    try:
        corpora = sel.resolve(tree)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    if not corpora:
        print("selection is empty; nothing to index", file=sys.stderr)
        return 1
    tok = load_tokenizer(cfg.model, cfg.tokenizer_path, allow_network=not args.offline)
    max_files = sel.max_files_per_corpus

    if args.dry_run:
        rows = []
        tf = tc = tt = 0
        for c in corpora:
            f, ch, t = dry_run_counts(tree, c, tok, cfg, max_files)
            tf, tc, tt = tf + f, tc + ch, tt + t
            rows.append((c.path, f"{f:,}", f"{ch:,}", f"{t:,}"))
        _print_table(rows, ("corpus", "files", "chunks", "tokens"))
        print(f"\n{len(corpora)} corpora · {tf:,} files · {tc:,} chunks · {tt:,} tokens · "
              f"est. {human_duration(tc / cfg.chunks_per_second)} (tokenizer: {tok.name})")
        return 0

    client = TEIClient(cfg.tei_url)
    info = client.info()
    check_model(info, cfg.model, cfg.dims)
    store = _open_store(cfg)
    try:
        store.migrate()
        store.ensure_meta(cfg.model, tok.name, cfg.chunk_tokens)
        measured = store.get_meta().get("chunks_per_second")
        if measured:
            cfg.chunks_per_second = float(measured)
        plans = []
        for c in corpora:
            pl = plan_corpus(tree, c, store, max_files)
            plans.append(pl)
            print(f"{c.path}: {len(pl.to_index):,} to index, {pl.unchanged:,} unchanged, "
                  f"{len(pl.to_delete):,} to delete")
        if not any(pl.to_index or pl.to_delete for pl in plans):
            print("nothing to do")
        else:
            def show(p: Progress) -> None:
                rate = p.chunks_per_second
                eta = ((p.files_total - p.files_done) * 1.05 / rate) if rate else 0
                print(f"\r{p.corpus:<48} files {p.files_done:,}/{p.files_total:,}  "
                      f"chunks {p.chunks_done:,}  {rate:.1f}/s  eta {human_duration(eta)}   ",
                      end="", file=sys.stderr, flush=True)

            indexer = Indexer(cfg, tree, store, client, tok, progress=show)
            p = indexer.run(plans)
            print(file=sys.stderr)
            print(f"indexed {p.files_done:,} files, {p.chunks_done:,} chunks in "
                  f"{human_duration(p.elapsed)} ({p.chunks_per_second:.1f} chunks/s)")
        if args.build_index or (not args.no_build_index and not store.has_hnsw()):
            print("building HNSW index…", file=sys.stderr)
            store.build_hnsw()
    finally:
        store.close()
    return 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    tree = _tree(cfg)
    sel = Selection.load(cfg.selection_file)
    try:
        store = _open_store(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"database unreachable: {exc}", file=sys.stderr)
        return 1
    try:
        store.migrate()
        meta = store.get_meta()
        rows_db = store.corpus_rows()
    finally:
        store.close()
    rows = []
    for c in tree.corpora:
        r = rows_db.get(c.path)
        if r is None:
            if not sel.is_selected(c) and not args.all:
                continue
            state = "not indexed"
        elif r.completed_at is None:
            state = "partial"
        elif r.commit_sha != c.commit_sha:
            state = "stale"
        else:
            state = "indexed " + r.completed_at.strftime("%Y-%m-%d")
        rows.append(("x" if sel.is_selected(c) else " ", c.path, state,
                     f"{r.documents:,}" if r else "", f"{r.chunks:,}" if r else ""))
    _print_table(rows, ("sel", "corpus", "state", "docs", "chunks"))
    if meta:
        print(f"\nindex: model={meta.get('model')} dims={meta.get('dims')} "
              f"type={meta.get('vector_type')} chunks/s={meta.get('chunks_per_second', '?')}")
    return 0


def cmd_search(cfg: Config, args: argparse.Namespace) -> int:
    from .embed import TEIClient
    from .search import search

    store = _open_store(cfg)
    try:
        hits = search(store, TEIClient(cfg.tei_url), args.query, query_prefix=cfg.query_prefix,
                      top=args.top, countries=args.country or [], corpora=args.corpus or [])
    finally:
        store.close()
    for i, h in enumerate(hits, 1):
        where = f"{h.path}" + (f" #{h.ordinal}" if h.ordinal else "")
        print(f"{i:2d}. {h.score:.3f}  {where}")
        print(f"    {h.title or ''}" + (f"  [{h.citation}]" if h.citation else "")
              + (f"  ({h.heading_path})" if h.heading_path else ""))
        if not args.short:
            snippet = " ".join(h.text.split())
            print(f"    {snippet[:args.width]}{'…' if len(snippet) > args.width else ''}")
    return 0


def cmd_reset(cfg: Config, args: argparse.Namespace) -> int:
    if not args.yes:
        answer = input(f"drop schema {cfg.schema!r} and every vector in it? type 'yes': ")
        if answer.strip().lower() != "yes":
            print("aborted")
            return 1
    store = _open_store(cfg)
    try:
        store.reset()
    finally:
        store.close()
    print("index dropped")
    return 0


def cmd_tui(cfg: Config, args: argparse.Namespace) -> int:
    try:
        from .tui.app import run_tui
    except ImportError:
        print("the TUI needs Textual: pip install 'global-law-vector[tui]'", file=sys.stderr)
        return 1
    return run_tui(cfg)


# --- parser ------------------------------------------------------------------------------------


def _add_selection_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--country", action="append", metavar="NAME", help="include a country")
    p.add_argument("--corpus", action="append", metavar="PATH", help="include a corpus path")
    p.add_argument("--exclude", action="append", metavar="NAME|PATH",
                   help="exclude a country or corpus")
    p.add_argument("--all", action="store_true", help="every pinned corpus")
    p.add_argument("--max-files", type=int, default=None, metavar="N",
                   help="sample at most N files per corpus (0 = no limit)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glv", description="global-law-vector: vectorised law search")
    p.add_argument("--version", action="version", version=f"glv {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="choose countries and corpora; show the cost")
    _add_selection_flags(s)
    s.add_argument("--clear", action="store_true", help="start from an empty selection")
    s.add_argument("--show", action="store_true", help="print the resolved selection with stats")
    s.add_argument("--paths", action="store_true", help="with --show: only the corpus paths")
    s.set_defaults(func=cmd_select)

    s = sub.add_parser("list", help="every corpus in the tree, marking the selected ones")
    s.add_argument("--country", action="append", metavar="NAME")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("index", help="index the selection (or a one-off selection)")
    _add_selection_flags(s)
    s.add_argument("--dry-run", action="store_true", help="parse and chunk, embed nothing")
    s.add_argument("--offline", action="store_true", help="do not fetch the tokenizer")
    s.add_argument("--build-index", action="store_true", help="(re)build the HNSW index after")
    s.add_argument("--no-build-index", action="store_true", help="skip the HNSW build")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("status", help="per corpus: selected, indexed, stale, partial")
    s.add_argument("--all", action="store_true", help="also list unselected, unindexed corpora")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("search", help="semantic search over the index")
    s.add_argument("query")
    s.add_argument("--country", action="append", metavar="NAME")
    s.add_argument("--corpus", action="append", metavar="PATH")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--short", action="store_true", help="no text snippet")
    s.add_argument("--width", type=int, default=240, help="snippet width")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("reset", help="drop the index schema")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_reset)

    s = sub.add_parser("tui", help="interactive terminal UI")
    s.set_defaults(func=cmd_tui)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
    cfg = Config.from_env()
    try:
        return args.func(cfg, args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
