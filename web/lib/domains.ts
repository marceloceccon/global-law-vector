/**
 * Eight fixed legal domains for colouring, chosen for the themes that dominate the indexed
 * corpora (statute books of ten civil- and common-law countries). Nothing is stored: each
 * sampled chunk is coloured by the domain whose vector it is closest to (see points.ts).
 *
 * `phrase` is embedded with the same model as the corpus (query side, so it gets the model's
 * "query: " prefix). `keywords` are only used when the embeddings server is unreachable: the
 * average embedding of chunks whose text mentions them stands in for the phrase vector.
 *
 * Colours are the dark-mode categorical slots of the dataviz reference palette, validated
 * for colour-vision deficiency in this order (the order the legend uses) against #04060d,
 * a slightly darker background than the page's current #0b0d12.
 */
export type Domain = {
  id: string;
  label: string;
  color: string;
  phrase: string;
  keywords: string[];
};

export const DOMAINS: readonly Domain[] = [
  {
    id: "contract",
    label: "Contract / Commercial",
    color: "#3987e5",
    phrase:
      "Contracts, obligations, sales, companies, commercial transactions and business law",
    keywords: ["contract", "contrat", "contrato", "vertrag", "договор", "commercial", "comercial",
      "société", "sociedade", "sociedad", "gesellschaft", "company", "obligation", "obrigação"],
  },
  {
    id: "criminal",
    label: "Criminal / Penal",
    color: "#d95926",
    phrase: "Criminal offences, penalties, imprisonment, prosecution and penal procedure",
    keywords: ["criminal", "crime", "penal", "pénal", "delito", "strafe", "strafbar", "преступ",
      "уголовн", "imprisonment", "prisão", "prisión", "emprisonnement", "freiheitsstrafe"],
  },
  {
    id: "administrative",
    label: "Administrative / Public",
    color: "#199e70",
    phrase:
      "Public administration, government bodies, civil servants, municipalities, licences and permits",
    keywords: ["administrat", "administração", "administración", "verwaltung", "администрац",
      "municipal", "servidor público", "funcionario", "fonctionnaire", "beamte", "public authority"],
  },
  {
    id: "tax",
    label: "Tax",
    color: "#c98500",
    phrase: "Taxes, duties, tax base, tax rates, assessment, collection and tax exemptions",
    keywords: ["tax", "impôt", "imposto", "impuesto", "tribut", "steuer", "налог", "fiscal",
      "contribuinte", "contribuyente", "taxpayer"],
  },
  {
    id: "family",
    label: "Family / Divorce",
    color: "#d55181",
    phrase: "Marriage, divorce, spouses, child custody, parental responsibility and alimony",
    keywords: ["divorce", "divórcio", "divorcio", "marriage", "mariage", "casamento", "matrimonio",
      "ehegatte", "scheidung", "брак", "развод", "custody", "guarda", "alimony", "pensão alimentícia",
      "époux", "cônjuge"],
  },
  {
    id: "labor",
    label: "Labor / Employment",
    color: "#008300",
    phrase: "Employment contracts, workers, wages, working hours, dismissal and trade unions",
    keywords: ["employ", "labour", "labor", "travail", "trabalho", "trabajo", "arbeitnehmer",
      "arbeitsvertrag", "труд", "salário", "salario", "salaire", "lohn", "worker", "trabajador",
      "trabalhador", "sindicato", "syndicat"],
  },
  {
    id: "inheritance",
    label: "Inheritance / Succession",
    color: "#9085e9",
    phrase: "Inheritance, succession, heirs, wills, testaments, estates and legacies",
    keywords: ["inheritance", "succession", "herança", "herencia", "sucesión", "sucessão", "erbe",
      "erbschaft", "testament", "наследств", "heir", "herdeiro", "heredero", "héritier", "legado"],
  },
  {
    id: "aml",
    label: "Anti-Money Laundering / Compliance",
    color: "#e66767",
    phrase:
      "Money laundering, terrorist financing, suspicious transactions, beneficial owners, sanctions and compliance obligations",
    keywords: ["money laundering", "blanchiment", "lavagem de dinheiro", "blanqueo de capitales",
      "geldwäsche", "отмыван", "terrorist financing", "financement du terrorisme",
      "beneficial owner", "beneficiário efetivo", "compliance", "suspicious transaction"],
  },
];
