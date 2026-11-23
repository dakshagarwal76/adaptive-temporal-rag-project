"""Generate timestamped corpus + labeled temporal queries for Adaptive Temporal RAG.

Target sizes (plan):
  - corpus: 600–1000 dated passages
  - queries: 800–1200 labeled (70/15/15), with train paraphrases
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_schema import CorpusDoc, QueryItem, write_jsonl

# ---------------------------------------------------------------------------
# Entity timelines: same entity, multiple dated facts
# ---------------------------------------------------------------------------

OFFICE_HOLDERS = [
    {
        "entity": "Prime Minister of India",
        "topic": "politics",
        "holders": [
            ("Manmohan Singh", "2004-05-22", "2014-05-25"),
            ("Narendra Modi", "2014-05-26", "2025-12-31"),
        ],
    },
    {
        "entity": "President of the United States",
        "topic": "politics",
        "holders": [
            ("Barack Obama", "2009-01-20", "2017-01-19"),
            ("Donald Trump", "2017-01-20", "2021-01-19"),
            ("Joe Biden", "2021-01-20", "2025-01-19"),
            ("Donald Trump", "2025-01-20", "2025-12-31"),
        ],
    },
    {
        "entity": "President of France",
        "topic": "politics",
        "holders": [
            ("François Hollande", "2012-05-15", "2017-05-13"),
            ("Emmanuel Macron", "2017-05-14", "2025-12-31"),
        ],
    },
    {
        "entity": "Chancellor of Germany",
        "topic": "politics",
        "holders": [
            ("Angela Merkel", "2005-11-22", "2021-12-07"),
            ("Olaf Scholz", "2021-12-08", "2025-05-05"),
            ("Friedrich Merz", "2025-05-06", "2025-12-31"),
        ],
    },
    {
        "entity": "Prime Minister of the United Kingdom",
        "topic": "politics",
        "holders": [
            ("David Cameron", "2010-05-11", "2016-07-12"),
            ("Theresa May", "2016-07-13", "2019-07-23"),
            ("Boris Johnson", "2019-07-24", "2022-09-05"),
            ("Liz Truss", "2022-09-06", "2022-10-24"),
            ("Rishi Sunak", "2022-10-25", "2024-07-04"),
            ("Keir Starmer", "2024-07-05", "2025-12-31"),
        ],
    },
    {
        "entity": "UN Secretary-General",
        "topic": "politics",
        "holders": [
            ("Ban Ki-moon", "2007-01-01", "2016-12-31"),
            ("António Guterres", "2017-01-01", "2025-12-31"),
        ],
    },
    {
        "entity": "Chief Justice of India",
        "topic": "politics",
        "holders": [
            ("N. V. Ramana", "2021-04-24", "2022-08-26"),
            ("U. U. Lalit", "2022-08-27", "2022-11-08"),
            ("D. Y. Chandrachud", "2022-11-09", "2024-11-10"),
            ("Sanjiv Khanna", "2024-11-11", "2025-05-13"),
            ("B. R. Gavai", "2025-05-14", "2025-12-31"),
        ],
    },
    {
        "entity": "President of Brazil",
        "topic": "politics",
        "holders": [
            ("Dilma Rousseff", "2011-01-01", "2016-08-31"),
            ("Michel Temer", "2016-09-01", "2018-12-31"),
            ("Jair Bolsonaro", "2019-01-01", "2022-12-31"),
            ("Luiz Inácio Lula da Silva", "2023-01-01", "2025-12-31"),
        ],
    },
]

CEOS = [
    {
        "entity": "CEO of Microsoft",
        "topic": "business",
        "holders": [
            ("Steve Ballmer", "2000-01-13", "2014-02-03"),
            ("Satya Nadella", "2014-02-04", "2025-12-31"),
        ],
    },
    {
        "entity": "CEO of Apple",
        "topic": "business",
        "holders": [
            ("Steve Jobs", "1997-09-16", "2011-08-24"),
            ("Tim Cook", "2011-08-25", "2025-12-31"),
        ],
    },
    {
        "entity": "CEO of Alphabet",
        "topic": "business",
        "holders": [
            ("Larry Page", "2015-08-10", "2019-12-03"),
            ("Sundar Pichai", "2019-12-04", "2025-12-31"),
        ],
    },
    {
        "entity": "CEO of Tesla",
        "topic": "business",
        "holders": [
            ("Elon Musk", "2008-10-01", "2025-12-31"),
        ],
    },
    {
        "entity": "CEO of Amazon",
        "topic": "business",
        "holders": [
            ("Jeff Bezos", "1994-07-05", "2021-07-04"),
            ("Andy Jassy", "2021-07-05", "2025-12-31"),
        ],
    },
    {
        "entity": "CEO of Twitter / X",
        "topic": "business",
        "holders": [
            ("Jack Dorsey", "2015-09-30", "2021-11-29"),
            ("Parag Agrawal", "2021-11-30", "2022-10-27"),
            ("Elon Musk", "2022-10-28", "2023-05-11"),
            ("Linda Yaccarino", "2023-05-12", "2025-12-31"),
        ],
    },
    {
        "entity": "CEO of OpenAI",
        "topic": "business",
        "holders": [
            ("Sam Altman", "2019-03-01", "2023-11-16"),
            ("Mira Murati", "2023-11-17", "2023-11-21"),
            ("Sam Altman", "2023-11-22", "2025-12-31"),
        ],
    },
    {
        "entity": "CEO of Meta",
        "topic": "business",
        "holders": [
            ("Mark Zuckerberg", "2004-02-04", "2025-12-31"),
        ],
    },
]

SPORTS = [
    {
        "entity": "FIFA World Cup winners",
        "topic": "sports",
        "events": [
            ("Germany", "2014-07-13", "Germany won the 2014 FIFA World Cup, defeating Argentina 1–0 in the final in Brazil."),
            ("France", "2018-07-15", "France won the 2018 FIFA World Cup, defeating Croatia 4–2 in the final in Russia."),
            ("Argentina", "2022-12-18", "Argentina won the 2022 FIFA World Cup, defeating France on penalties in Qatar."),
        ],
    },
    {
        "entity": "Wimbledon men's singles champion",
        "topic": "sports",
        "events": [
            ("Novak Djokovic", "2019-07-14", "Novak Djokovic won the 2019 Wimbledon men's singles title."),
            ("Novak Djokovic", "2021-07-11", "Novak Djokovic won the 2021 Wimbledon men's singles title."),
            ("Novak Djokovic", "2022-07-10", "Novak Djokovic won the 2022 Wimbledon men's singles title."),
            ("Carlos Alcaraz", "2023-07-16", "Carlos Alcaraz won the 2023 Wimbledon men's singles title."),
            ("Carlos Alcaraz", "2024-07-14", "Carlos Alcaraz won the 2024 Wimbledon men's singles title."),
        ],
    },
    {
        "entity": "ICC Cricket World Cup winners",
        "topic": "sports",
        "events": [
            ("Australia", "2015-03-29", "Australia won the 2015 ICC Cricket World Cup, defeating New Zealand in the final."),
            ("England", "2019-07-14", "England won the 2019 ICC Cricket World Cup after a dramatic Super Over finish against New Zealand."),
            ("Australia", "2023-11-19", "Australia won the 2023 ICC Cricket World Cup, defeating India in the final in Ahmedabad."),
        ],
    },
    {
        "entity": "UEFA Champions League winners",
        "topic": "sports",
        "events": [
            ("Real Madrid", "2018-05-26", "Real Madrid won the 2017–18 UEFA Champions League, defeating Liverpool 3–1 in Kyiv."),
            ("Liverpool", "2019-06-01", "Liverpool won the 2018–19 UEFA Champions League, defeating Tottenham 2–0 in Madrid."),
            ("Bayern Munich", "2020-08-23", "Bayern Munich won the 2019–20 UEFA Champions League, defeating Paris Saint-Germain 1–0."),
            ("Chelsea", "2021-05-29", "Chelsea won the 2020–21 UEFA Champions League, defeating Manchester City 1–0."),
            ("Real Madrid", "2022-05-28", "Real Madrid won the 2021–22 UEFA Champions League, defeating Liverpool 1–0."),
            ("Manchester City", "2023-06-10", "Manchester City won the 2022–23 UEFA Champions League, defeating Inter Milan 1–0."),
            ("Real Madrid", "2024-06-01", "Real Madrid won the 2023–24 UEFA Champions League, defeating Borussia Dortmund 2–0."),
        ],
    },
    {
        "entity": "Formula 1 World Drivers' Champion",
        "topic": "sports",
        "events": [
            ("Lewis Hamilton", "2018-11-25", "Lewis Hamilton was crowned the 2018 Formula 1 World Drivers' Champion."),
            ("Lewis Hamilton", "2019-11-03", "Lewis Hamilton was crowned the 2019 Formula 1 World Drivers' Champion."),
            ("Lewis Hamilton", "2020-11-15", "Lewis Hamilton was crowned the 2020 Formula 1 World Drivers' Champion."),
            ("Max Verstappen", "2021-12-12", "Max Verstappen won the 2021 Formula 1 World Drivers' Championship."),
            ("Max Verstappen", "2022-10-09", "Max Verstappen won the 2022 Formula 1 World Drivers' Championship."),
            ("Max Verstappen", "2023-10-07", "Max Verstappen won the 2023 Formula 1 World Drivers' Championship."),
            ("Max Verstappen", "2024-12-08", "Max Verstappen won the 2024 Formula 1 World Drivers' Championship."),
        ],
    },
    {
        "entity": "Olympic Games host city",
        "topic": "sports",
        "events": [
            ("Rio de Janeiro", "2016-08-05", "The 2016 Summer Olympics were hosted by Rio de Janeiro, Brazil."),
            ("Tokyo", "2021-07-23", "The 2020 Summer Olympics were hosted by Tokyo, Japan (held in 2021)."),
            ("Paris", "2024-07-26", "The 2024 Summer Olympics were hosted by Paris, France."),
        ],
    },
]

AWARDS = [
    {
        "entity": "Nobel Peace Prize",
        "topic": "awards",
        "events": [
            ("Malala Yousafzai and Kailash Satyarthi", "2014-10-10", "The 2014 Nobel Peace Prize was awarded to Malala Yousafzai and Kailash Satyarthi."),
            ("Juan Manuel Santos", "2016-10-07", "The 2016 Nobel Peace Prize was awarded to Juan Manuel Santos."),
            ("World Food Programme", "2020-10-09", "The 2020 Nobel Peace Prize was awarded to the World Food Programme."),
            ("Maria Ressa and Dmitry Muratov", "2021-10-08", "The 2021 Nobel Peace Prize was awarded to Maria Ressa and Dmitry Muratov."),
            ("Ales Bialiatski, Memorial, and Center for Civil Liberties", "2022-10-07", "The 2022 Nobel Peace Prize was shared by Ales Bialiatski, Memorial, and the Center for Civil Liberties."),
            ("Narges Mohammadi", "2023-10-06", "The 2023 Nobel Peace Prize was awarded to Narges Mohammadi."),
            ("Nihon Hidankyo", "2024-10-11", "The 2024 Nobel Peace Prize was awarded to Nihon Hidankyo."),
        ],
    },
    {
        "entity": "Academy Award for Best Picture",
        "topic": "awards",
        "events": [
            ("Parasite", "2020-02-09", "Parasite won the Academy Award for Best Picture at the 92nd Oscars."),
            ("Nomadland", "2021-04-25", "Nomadland won the Academy Award for Best Picture at the 93rd Oscars."),
            ("CODA", "2022-03-27", "CODA won the Academy Award for Best Picture at the 94th Oscars."),
            ("Everything Everywhere All at Once", "2023-03-12", "Everything Everywhere All at Once won Best Picture at the 95th Oscars."),
            ("Oppenheimer", "2024-03-10", "Oppenheimer won the Academy Award for Best Picture at the 96th Oscars."),
        ],
    },
    {
        "entity": "Ballon d'Or winner",
        "topic": "awards",
        "events": [
            ("Lionel Messi", "2019-12-02", "Lionel Messi won the 2019 Ballon d'Or."),
            ("Robert Lewandowski", "2021-11-29", "Robert Lewandowski was widely recognized as the top player of 2020–21; Messi won the 2021 Ballon d'Or."),
            ("Lionel Messi", "2021-11-29", "Lionel Messi won the 2021 Ballon d'Or."),
            ("Karim Benzema", "2022-10-17", "Karim Benzema won the 2022 Ballon d'Or."),
            ("Lionel Messi", "2023-10-30", "Lionel Messi won the 2023 Ballon d'Or."),
            ("Rodri", "2024-10-28", "Rodri won the 2024 Ballon d'Or."),
        ],
    },
]

LAWS_POLICIES = [
    {
        "entity": "GST in India",
        "topic": "law",
        "events": [
            ("Introduced", "2017-07-01", "India's Goods and Services Tax (GST) came into effect on 1 July 2017, replacing multiple indirect taxes."),
            ("Rate revision note", "2022-12-01", "As of late 2022, GST continued as India's principal indirect tax with periodic GST Council rate revisions."),
            ("central indirect tax framework", "2024-06-15", "In 2024, GST remained the central indirect tax framework administered through the GST Council in India."),
        ],
    },
    {
        "entity": "GDPR",
        "topic": "law",
        "events": [
            ("Enforced", "2018-05-25", "The EU General Data Protection Regulation (GDPR) became enforceable on 25 May 2018."),
            ("Enforcement active", "2021-06-01", "By 2021, GDPR enforcement was active across the EU with significant fines for non-compliance."),
            ("Still in force", "2024-05-25", "As of 2024, GDPR remained the core EU data protection regulation for personal data processing."),
        ],
    },
    {
        "entity": "Article 370 of the Indian Constitution",
        "topic": "law",
        "events": [
            ("special autonomous status", "2018-01-01", "Before August 2019, Article 370 granted special autonomous status to Jammu and Kashmir."),
            ("Abrogated", "2019-08-05", "On 5 August 2019, the Government of India revoked the special status of Jammu and Kashmir under Article 370."),
            ("UT reorganization", "2019-10-31", "On 31 October 2019, under Article 370 of the Indian Constitution, Jammu and Kashmir was reorganized into two Union Territories (UT reorganization): Jammu & Kashmir and Ladakh."),
        ],
    },
]

TIMELESS_FACTS = [
    ("Photosynthesis", "biology", "Photosynthesis is the process by which green plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water."),
    ("Pythagorean theorem", "math", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides."),
    ("DNA", "biology", "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for the development and functioning of living organisms."),
    ("Gravity", "physics", "Gravity is a fundamental force of attraction between masses; near Earth's surface it gives objects weight."),
    ("HTTP", "technology", "HTTP (Hypertext Transfer Protocol) is the application-layer protocol used for transferring hypermedia documents on the World Wide Web."),
    ("Compound interest", "finance", "Compound interest is interest calculated on the initial principal and also on the accumulated interest from previous periods."),
    ("Supply and demand", "economics", "Supply and demand is an economic model describing how prices are determined in a competitive market by the interaction of sellers and buyers."),
    ("Binary search", "computer science", "Binary search is an algorithm that finds a target value in a sorted array by repeatedly dividing the search interval in half."),
    ("Osmosis", "biology", "Osmosis is the net movement of solvent molecules through a selectively permeable membrane from low to high solute concentration."),
    ("Prime number", "math", "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."),
    ("Entropy", "physics", "In thermodynamics, entropy is a measure of the disorder or randomness of a system."),
    ("TCP/IP", "technology", "TCP/IP is a suite of communication protocols used to interconnect network devices on the Internet."),
    ("Machine learning", "computer science", "Machine learning is a field of artificial intelligence in which systems learn patterns from data to make predictions or decisions."),
    ("Inflation", "economics", "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power."),
    ("Photosynthetic equation", "biology", "The simplified overall equation for photosynthesis is 6CO2 + 6H2O + light → C6H12O6 + 6O2."),
    ("Hash table", "computer science", "A hash table is a data structure that maps keys to values using a hash function for average-case constant-time lookup."),
    ("Newton's second law", "physics", "Newton's second law states that force equals mass times acceleration (F = ma)."),
    ("GDP", "economics", "Gross Domestic Product (GDP) is the total monetary value of all finished goods and services produced within a country in a given period."),
    ("Recursion", "computer science", "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem."),
    ("Mitochondria", "biology", "Mitochondria are organelles that generate most of the cell's supply of ATP through cellular respiration."),
    ("Bayes' theorem", "math", "Bayes' theorem describes how to update the probability of a hypothesis given new evidence."),
    ("API", "technology", "An API (Application Programming Interface) is a set of definitions and protocols for building and integrating application software."),
    ("Opportunity cost", "economics", "Opportunity cost is the value of the next-best alternative forgone when a choice is made."),
    ("Stack data structure", "computer science", "A stack is a last-in, first-out (LIFO) data structure supporting push and pop operations."),
    ("Relativity (special)", "physics", "Special relativity, introduced by Einstein, describes the physics of objects moving at constant velocity relative to each other, near light speed."),
    ("Enzyme", "biology", "Enzymes are biological catalysts that speed up chemical reactions in living organisms without being consumed."),
    ("Derivative (calculus)", "math", "A derivative measures the instantaneous rate of change of a function with respect to one of its variables."),
    ("Cloud computing", "technology", "Cloud computing is the on-demand delivery of computing resources over the Internet with pay-as-you-go pricing."),
    ("Comparative advantage", "economics", "Comparative advantage is the ability of a party to produce a good or service at a lower opportunity cost than another."),
    ("Graph traversal BFS", "computer science", "Breadth-first search (BFS) explores a graph level by level, visiting all neighbors of a node before moving deeper."),
]


TEMPLATES_CURRENT = [
    "Who is the current {entity}?",
    "Who currently holds the position of {entity}?",
    "As of now, who is the {entity}?",
    "Name the present {entity}.",
    "Who is serving as {entity} today?",
]

TEMPLATES_HISTORICAL = [
    "Who was the {entity} in {year}?",
    "In {year}, who held the position of {entity}?",
    "Who served as {entity} during {year}?",
    "As of {date}, who was the {entity}?",
    "Name the {entity} in the year {year}.",
]

TEMPLATES_EVENT_CURRENT = [
    "Who won the most recent {entity}?",
    "What is the latest result for {entity}?",
    "Who is the current / most recent {entity}?",
]

TEMPLATES_EVENT_HIST = [
    "Who won the {entity} in {year}?",
    "What was the outcome of {entity} around {year}?",
    "In {year}, who or what was associated with {entity}?",
]

TEMPLATES_TIMELESS = [
    "What is {entity}?",
    "Define {entity}.",
    "Explain {entity} briefly.",
    "Give a short definition of {entity}.",
]

PARAPHRASE_PREFIXES = [
    "Please tell me: ",
    "Quick question — ",
    "I need to know: ",
    "",
    "Can you answer: ",
]


def _parse(d: str) -> date:
    y, m, day = map(int, d.split("-"))
    return date(y, m, day)


def _midpoint(start: str, end: str) -> str:
    a, b = _parse(start), _parse(end)
    mid = a + (b - a) / 2
    return mid.isoformat()


def _year_sample(start: str, end: str, rng: random.Random) -> tuple[str, str]:
    a, b = _parse(start), _parse(end)
    if b <= a:
        return start, str(a.year)
    delta = (b - a).days
    offset = rng.randint(0, max(delta, 0))
    d = a + timedelta(days=offset)
    return d.isoformat(), str(d.year)


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.md5("|".join(parts).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


# Fixed current snapshots inside the eval's 3-year window (reference date 2025-10-01).
# Random tenure sampling does not guarantee a recent passage for every entity.
REFERENCE_SNAPSHOTS = [
    (
        "Prime Minister of India",
        "Narendra Modi",
        "politics",
        "As of 2025-10-01, Narendra Modi is the Prime Minister of India.",
    ),
    (
        "CEO of Tesla",
        "Elon Musk",
        "business",
        "As of 2025-10-01, Elon Musk is the CEO of Tesla.",
    ),
    (
        "Article 370 of the Indian Constitution",
        "UT reorganization",
        "law",
        "As of 2025-10-01, the latest recorded outcome for Article 370 of the Indian Constitution is the UT reorganization of Jammu and Kashmir and Ladakh.",
    ),
]


def reference_snapshot_docs() -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    ts = "2025-10-01"
    for entity, name, topic, text in REFERENCE_SNAPSHOTS:
        docs.append(
            CorpusDoc(
                doc_id=_stable_id("doc", entity, name, ts, "reference"),
                text=text,
                timestamp=ts,
                title=f"{entity} — {name}",
                source="synthetic_timeline",
                entity=entity,
                topic=topic,
            )
        )
    return docs


def _prefer_reference_snapshot(docs: list[CorpusDoc], entity: str, gold_ids: list[str]) -> list[str]:
    """Put the 2025-10-01 reference passage in a current gold set and drop the oldest id."""
    refs = [d for d in reference_snapshot_docs() if d.entity == entity]
    if not refs:
        return gold_ids
    ref_id = refs[0].doc_id
    if ref_id in gold_ids:
        return gold_ids[:3]
    by_id = {d.doc_id: d for d in docs}
    kept = [i for i in gold_ids if i != ref_id]
    if len(kept) >= 3:
        oldest = min(kept, key=lambda i: (by_id[i].timestamp if i in by_id else "", i))
        kept = [i for i in kept if i != oldest]
    return [ref_id, *kept][:3]


def build_corpus(rng: random.Random) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []

    # Office holders — multiple snapshots per tenure
    for block in OFFICE_HOLDERS + CEOS:
        entity = block["entity"]
        topic = block["topic"]
        holders = block["holders"]
        for name, start, end in holders:
            # 2–4 dated snapshots per tenure
            n_snap = 3 if start != end else 2
            for i in range(n_snap):
                ts, _ = _year_sample(start, end, rng)
                variants = [
                    f"As of {ts}, {name} is the {entity}.",
                    f"On {ts}, records list {name} as the serving {entity}.",
                    f"{name} held the office of {entity} as reported around {ts}.",
                    f"Official and news sources around {ts} identify {name} as the {entity}.",
                ]
                text = rng.choice(variants)
                docs.append(
                    CorpusDoc(
                        doc_id=_stable_id("doc", entity, name, ts, text[:40]),
                        text=text,
                        timestamp=ts,
                        title=f"{entity} — {name}",
                        source="synthetic_timeline",
                        entity=entity,
                        topic=topic,
                    )
                )
            # Boundary announcement docs
            docs.append(
                CorpusDoc(
                    doc_id=_stable_id("doc", entity, name, start, "start"),
                    text=f"On {start}, {name} assumed office as the {entity}.",
                    timestamp=start,
                    title=f"{entity} appointment — {name}",
                    source="synthetic_timeline",
                    entity=entity,
                    topic=topic,
                )
            )

    for block in SPORTS + AWARDS + LAWS_POLICIES:
        entity = block["entity"]
        topic = block["topic"]
        for name, ts, text in block["events"]:
            docs.append(
                CorpusDoc(
                    doc_id=_stable_id("doc", entity, name, ts),
                    text=text,
                    timestamp=ts,
                    title=f"{entity} — {name}",
                    source="synthetic_timeline",
                    entity=entity,
                    topic=topic,
                )
            )
            # Echo passage with slight rewording for denser corpus
            echo = f"Dated {ts}: {text}"
            docs.append(
                CorpusDoc(
                    doc_id=_stable_id("doc", entity, name, ts, "echo"),
                    text=echo,
                    timestamp=ts,
                    title=f"{entity} note — {name}",
                    source="synthetic_timeline",
                    entity=entity,
                    topic=topic,
                )
            )

    # Timeless encyclopedia-style docs (dated as reference snapshots, but facts don't change)
    for title, topic, text in TIMELESS_FACTS:
        for year in (2016, 2019, 2022, 2025):
            ts = f"{year}-01-15"
            docs.append(
                CorpusDoc(
                    doc_id=_stable_id("doc", title, ts),
                    text=text,
                    timestamp=ts,
                    title=title,
                    source="synthetic_encyclopedia",
                    entity=title,
                    topic=topic,
                )
            )

    # Extra filler dated news-style blurbs to reach ~600–1000
    filler_topics = [
        ("Sensex closing note", "markets", "The BSE Sensex closed mixed amid global cues and domestic institutional flows."),
        ("Monsoon update", "weather", "The India Meteorological Department issued a routine monsoon progress update for the season."),
        ("Startup funding round", "business", "A technology startup announced a new funding round to expand product development and hiring."),
        ("Space mission update", "science", "Space agencies published a status update on an ongoing orbital science mission."),
        ("Public health advisory", "health", "Health authorities reiterated seasonal vaccination and hygiene advisories for the public."),
        ("Railway timetable revision", "transport", "Indian Railways announced selective timetable revisions on selected routes."),
        ("University ranking note", "education", "An annual university ranking report highlighted research output and teaching metrics."),
        ("Climate summit summary", "environment", "Delegates at a climate meeting summarized commitments on emissions and adaptation finance."),
    ]
    base = date(2015, 1, 1)
    while len(docs) < 800:
        title, topic, text = rng.choice(filler_topics)
        days = rng.randint(0, 365 * 10)
        ts = (base + timedelta(days=days)).isoformat()
        docs.append(
            CorpusDoc(
                doc_id=_stable_id("doc", title, ts, str(len(docs))),
                text=f"As of {ts}, {text}",
                timestamp=ts,
                title=title,
                source="synthetic_news",
                entity=title,
                topic=topic,
            )
        )

    # Cap at 1000
    if len(docs) > 1000:
        rng.shuffle(docs)
        docs = docs[:1000]
    present = {d.doc_id for d in docs}
    for snap in reference_snapshot_docs():
        if snap.doc_id not in present:
            docs.append(snap)
    docs.sort(key=lambda d: (d.timestamp, d.doc_id))
    return docs


def _holder_at(holders: list[tuple[str, str, str]], when: str) -> str | None:
    w = _parse(when)
    for name, start, end in holders:
        if _parse(start) <= w <= _parse(end):
            return name
    return None


def _docs_for_entity(docs: list[CorpusDoc], entity: str, when: str | None = None) -> list[str]:
    cands = [d for d in docs if d.entity == entity]
    if when:
        # prefer docs near the date
        w = _parse(when)
        cands = sorted(cands, key=lambda d: abs((_parse(d.timestamp) - w).days))
    return [d.doc_id for d in cands[:3]]


def build_queries(docs: list[CorpusDoc], rng: random.Random, target: int = 1000) -> list[QueryItem]:
    queries: list[QueryItem] = []
    qn = 0

    def add(
        text: str,
        needs: int,
        tclass: str,
        answer: str,
        gold_ids: list[str],
        time_hint: str | None = None,
        paraphrase: bool = False,
        seed: str | None = None,
    ) -> QueryItem:
        nonlocal qn
        qn += 1
        item = QueryItem(
            query_id=f"q_{qn:04d}",
            query=text,
            needs_retrieval=needs,
            temporal_class=tclass,  # type: ignore[arg-type]
            time_hint=time_hint,
            gold_answer=answer,
            gold_doc_ids=gold_ids,
            split="train",
            is_paraphrase=paraphrase,
            seed_query_id=seed,
        )
        queries.append(item)
        return item

    # --- Current / historical for office + CEO ---
    for block in OFFICE_HOLDERS + CEOS:
        entity = block["entity"]
        holders = block["holders"]
        current_name = holders[-1][0]
        recent = sorted([d for d in docs if d.entity == entity], key=lambda d: d.timestamp, reverse=True)
        gold_cur = _prefer_reference_snapshot(docs, entity, [d.doc_id for d in recent[:3]])

        # More current templates (repeat with mild wording) for class balance
        for tmpl in TEMPLATES_CURRENT:
            add(
                tmpl.format(entity=entity),
                needs=1,
                tclass="current",
                answer=current_name,
                gold_ids=gold_cur,
            )
            add(
                tmpl.format(entity=entity).replace("?", ""),
                needs=1,
                tclass="current",
                answer=current_name,
                gold_ids=gold_cur,
            )

        # One historical sample per tenure (not every template × tenure)
        for name, start, end in holders:
            ts, year = _year_sample(start, end, rng)
            who = _holder_at(holders, ts) or name
            gold = _docs_for_entity(docs, entity, ts)
            tmpl = rng.choice(TEMPLATES_HISTORICAL)
            add(
                tmpl.format(entity=entity, year=year, date=ts),
                needs=1,
                tclass="historical",
                answer=who,
                gold_ids=gold,
                time_hint=year,
            )

    # --- Sports / awards events ---
    for block in SPORTS + AWARDS + LAWS_POLICIES:
        entity = block["entity"]
        events = block["events"]
        latest_name, latest_ts, _ = events[-1]
        gold_cur = _prefer_reference_snapshot(docs, entity, _docs_for_entity(docs, entity, latest_ts))
        for tmpl in TEMPLATES_EVENT_CURRENT:
            add(
                tmpl.format(entity=entity),
                needs=1,
                tclass="current",
                answer=latest_name,
                gold_ids=gold_cur,
            )
            add(
                f"Currently, {tmpl.format(entity=entity).rstrip('?').lower()}?",
                needs=1,
                tclass="current",
                answer=latest_name,
                gold_ids=gold_cur,
            )
        for name, ts, _ in events:
            year = ts[:4]
            gold = _docs_for_entity(docs, entity, ts)
            tmpl = rng.choice(TEMPLATES_EVENT_HIST)
            add(
                tmpl.format(entity=entity, year=year),
                needs=1,
                tclass="historical",
                answer=name,
                gold_ids=gold,
                time_hint=year,
            )

    # --- Timeless (more no-retrieval to hit ~25–30%) ---
    for title, _topic, definition in TIMELESS_FACTS:
        gold = _docs_for_entity(docs, title)
        for tmpl in TEMPLATES_TIMELESS:
            needs = 1 if rng.random() < 0.35 else 0
            add(
                tmpl.format(entity=title),
                needs=needs,
                tclass="timeless",
                answer=definition.split(".")[0] + ".",
                gold_ids=gold if needs else [],
            )

    trivial = [
        ("What is 2 + 2?", "4"),
        ("What is 10 times 5?", "50"),
        ("How many days are in a week?", "7"),
        ("What color is a clear daytime sky typically described as?", "blue"),
        ("Spell the word cat.", "cat"),
        ("What is the sum of angles in a triangle in degrees?", "180"),
        ("Is water H2O?", "yes"),
        ("How many hours are in a day?", "24"),
        ("What is 7 + 8?", "15"),
        ("How many minutes are in an hour?", "60"),
        ("What is the opposite of hot?", "cold"),
        ("Is the sun a star?", "yes"),
    ]
    for q, a in trivial:
        for _ in range(4):
            add(q, needs=0, tclass="timeless", answer=a, gold_ids=[])

    # Class-balanced paraphrase fill to target
    by_class = {"current": [], "historical": [], "timeless": []}
    for q in queries:
        if not q.is_paraphrase:
            by_class[q.temporal_class].append(q)

    def next_seed(preferred: str) -> QueryItem:
        pool = by_class[preferred] or [q for q in queries if not q.is_paraphrase]
        return rng.choice(pool)

    # Target mix ~35/35/30 among final set
    while len(queries) < target:
        r = rng.random()
        preferred = "current" if r < 0.35 else ("historical" if r < 0.70 else "timeless")
        seed = next_seed(preferred)
        prefix = rng.choice(PARAPHRASE_PREFIXES)
        suffix_opts = ["", "?", " Please be brief.", " Answer briefly."]
        base = seed.query.rstrip("?")
        text = f"{prefix}{base}{rng.choice(suffix_opts)}".strip()
        if text == seed.query:
            text = f"Kindly answer: {seed.query}"
        add(
            text,
            needs=seed.needs_retrieval,
            tclass=seed.temporal_class,
            answer=seed.gold_answer,
            gold_ids=list(seed.gold_doc_ids),
            time_hint=seed.time_hint,
            paraphrase=True,
            seed=seed.query_id,
        )

    # Splits: 70/15/15 per temporal class on seed queries (balanced val/test); paraphrases -> train
    seeds = [q for q in queries if not q.is_paraphrase]
    from collections import defaultdict

    by_class: dict[str, list] = defaultdict(list)
    for q in seeds:
        by_class[q.temporal_class].append(q)
    for group in by_class.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        for idx, q in enumerate(group):
            if idx < n_train:
                q.split = "train"
            elif idx < n_train + n_val:
                q.split = "val"
            else:
                q.split = "test"
    seed_split = {q.query_id: q.split for q in seeds}
    for q in queries:
        if q.is_paraphrase:
            q.split = "train"
        elif q.query_id in seed_split:
            q.split = seed_split[q.query_id]

    # Re-number for stable ids
    for i, q in enumerate(queries, start=1):
        q.query_id = f"q_{i:04d}"

    return queries


def print_stats(docs: list[CorpusDoc], queries: list[QueryItem]) -> None:
    from collections import Counter

    print(f"Corpus size: {len(docs)}")
    years = Counter(d.timestamp[:4] for d in docs)
    print(f"  Year coverage: {dict(sorted(years.items()))}")
    print(f"  Unique entities: {len({d.entity for d in docs})}")

    print(f"Queries size: {len(queries)}")
    print(f"  By split: {dict(Counter(q.split for q in queries))}")
    print(f"  By temporal_class: {dict(Counter(q.temporal_class for q in queries))}")
    print(f"  needs_retrieval=0: {sum(1 for q in queries if q.needs_retrieval == 0)}")
    print(f"  needs_retrieval=1: {sum(1 for q in queries if q.needs_retrieval == 1)}")
    print(f"  paraphrases: {sum(1 for q in queries if q.is_paraphrase)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-queries",
        type=int,
        default=1000,
        help="Total labeled queries including paraphrases. "
        "T4 pilot ~2500–3500; A100 large run ~6000–8000.",
    )
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs = build_corpus(rng)
    queries = build_queries(docs, rng, target=args.target_queries)

    out = Path(args.out_dir)
    n_docs = write_jsonl(out / "corpus.jsonl", docs)
    n_q = write_jsonl(out / "queries.jsonl", queries)
    print_stats(docs, queries)
    print(f"Wrote {n_docs} docs -> {out / 'corpus.jsonl'}")
    print(f"Wrote {n_q} queries -> {out / 'queries.jsonl'}")


if __name__ == "__main__":
    main()
