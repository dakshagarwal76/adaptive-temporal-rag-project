"""Counterfactual minimal-pair augmentation for router_v2.
Entities and templates here were written for training only; every generated row is
screened against forbidden_v2 (exact, entity and template overlap with the eval sets)."""
import random, itertools

ORG_OFFICE = [  # (holder phrase, organisation)
 ("chief executive", "Nvidia"), ("chief executive", "Intel"), ("CEO", "Netflix"), ("CEO", "Boeing"),
 ("CEO", "Toyota"), ("CEO", "Siemens"), ("CEO", "Unilever"), ("CEO", "Pfizer"), ("CEO", "Shell plc"),
 ("CEO", "Volkswagen"), ("CEO", "IBM"), ("CEO", "Oracle"), ("CEO", "Sony"), ("CEO", "Nestle"),
 ("CEO", "Walmart"), ("CEO", "Coca-Cola"), ("CEO", "Disney+"), ("CEO", "Adidas"), ("CEO", "Infosys"),
 ("CEO", "Tata Motors"), ("CEO", "Reliance Industries"), ("CEO", "Spotify"), ("CEO", "Uber"),
 ("CEO", "Airbus"), ("CEO", "BP"), ("chair", "the World Bank"), ("director-general", "the WTO"),
 ("secretary general", "OPEC"), ("president", "the European Commission"), ("president", "the IOC"),
 ("president", "UEFA"), ("governor", "the Bank of England"), ("governor", "the Reserve Bank of India"),
 ("chair", "the European Central Bank board"), ("director", "the FBI"), ("director", "the CIA"),
 ("head", "Interpol"), ("president", "the African Union Commission"), ("secretary general", "the Commonwealth"),
 ("director-general", "UNESCO"), ("president", "the World Economic Forum"), ("executive director", "UNICEF"),
]
COUNTRY_OFFICE = [
 ("president", "Argentina"), ("president", "Mexico"), ("president", "South Korea"), ("president", "Kenya"),
 ("president", "Indonesia"), ("president", "Turkey"), ("president", "Egypt"), ("president", "Chile"),
 ("president", "Colombia"), ("president", "Ukraine"), ("president", "Poland"), ("president", "Portugal"),
 ("prime minister", "Spain"), ("prime minister", "Italy"), ("prime minister", "Canada"),
 ("prime minister", "New Zealand"), ("prime minister", "Pakistan"), ("prime minister", "Bangladesh"),
 ("prime minister", "Sweden"), ("prime minister", "Greece"), ("prime minister", "Israel"),
 ("prime minister", "Thailand"), ("prime minister", "Ireland"), ("prime minister", "Netherlands"),
 ("king", "Spain"), ("king", "Sweden"), ("monarch", "Denmark"), ("chancellor", "Austria"),
 ("foreign minister", "Brazil"), ("finance minister", "South Africa"), ("defence minister", "Australia"),
 ("speaker of parliament", "Sri Lanka"), ("chief minister", "Kerala"), ("governor", "California"),
 ("governor", "Texas"), ("mayor", "Tokyo"), ("mayor", "London"), ("mayor", "Mumbai"), ("mayor", "Berlin"),
]
METRIC_PLACE = [
 ("population", "Vietnam"), ("population", "Ethiopia"), ("population", "Brazil"), ("population", "Germany"),
 ("population", "Lagos"), ("population", "Shanghai"), ("GDP", "Italy"), ("GDP", "Saudi Arabia"),
 ("GDP growth rate", "China"), ("unemployment figure", "Spain"), ("inflation figure", "Turkey"),
 ("interest rate", "the eurozone"), ("national debt", "Greece"), ("minimum wage", "Germany"),
 ("price of gold", "the international market"), ("price of crude oil", "world markets"),
 ("share price", "Nvidia"), ("market capitalisation", "Saudi Aramco"), ("value", "the British pound"),
 ("average house price", "Sydney"), ("literacy rate", "Bihar"), ("life expectancy", "Russia"),
]
TITLES = [
 "the Premier League", "the NFL", "the Champions Trophy", "the Copa America", "the Ryder Cup",
 "the US Open men's title", "the Tour of Flanders", "the Six Nations", "the IPL", "La Liga",
 "the Davis Cup", "the America's Cup", "the world heavyweight boxing title", "the Ashes",
 "the Indy 500", "the Masters golf tournament", "the Africa Cup of Nations", "Serie A", "the Euroleague",
 "the World Series", "the French Open women's title", "the Asian Cup", "the Stanley Cup Final MVP award",
]
ANCHOR_EVENTS = [  # implicit past anchors
 "the fall of the Roman Empire", "the French Revolution", "the Great Depression", "the Second World War",
 "the First World War", "the moon landing", "the partition of India", "the Chernobyl disaster",
 "the dissolution of the Soviet Union", "the Korean War", "the Gulf War", "the Industrial Revolution",
 "the American Civil War", "the signing of the Treaty of Versailles", "the Spanish flu pandemic",
 "the 9/11 attacks", "the Watergate scandal", "the launch of Sputnik", "the Boston Tea Party",
 "the Meiji Restoration", "the Russian Revolution", "the Napoleonic Wars", "the Brexit referendum",
 "the Asian financial crisis", "the dot-com crash", "the Vietnam War", "the Iranian Revolution",
]
PREDECESSORS = [  # (office phrase, person) -> "before/after person"
 ("Amazon's chief executive", "Andy Jassy"), ("Google's CEO", "Sundar Pichai"), ("India's prime minister", "Manmohan Singh"),
 ("the US president", "Ronald Reagan"), ("Britain's prime minister", "Margaret Thatcher"),
 ("France's president", "Francois Mitterrand"), ("Russia's president", "Boris Yeltsin"),
 ("the Pope", "John Paul II"), ("South Africa's president", "Nelson Mandela"), ("China's leader", "Deng Xiaoping"),
 ("Canada's prime minister", "Pierre Trudeau"), ("Australia's prime minister", "John Howard"),
 ("Facebook's CEO", "Mark Zuckerberg"), ("Tesla's chief executive", "Martin Eberhard"),
 ("the UN Secretary-General", "Ban Ki-moon"), ("the England football manager", "Gareth Southgate"),
 ("Italy's prime minister", "Silvio Berlusconi"), ("Brazil's president", "Fernando Henrique Cardoso"),
]
FIRSTS = [
 "Nobel Prize in Physics", "Academy Award for Best Actress", "Rugby World Cup", "Cricket World Cup",
 "Super League title", "Grand Slam tennis tournament", "transatlantic flight", "Formula One race",
 "Tour of Italy", "Olympic marathon", "Booker of Bookers", "Turing Award", "Fields Medal",
]
STABLE_FACT = [
 "Who wrote {x}?", "Who composed {x}?", "Who painted {x}?", "Who discovered {x}?", "Who invented {x}?",
 "What is the main theme of {x}?", "In which country is {x} located?", "What is {x} made of?",
]
WORKS = [
 "Pride and Prejudice", "War and Peace", "the Odyssey", "Don Quixote", "The Magic Flute", "Swan Lake",
 "the Mona Lisa", "Guernica", "The Starry Night", "penicillin", "the telephone", "the printing press",
 "radium", "the Eiffel Tower", "Machu Picchu", "the Taj Mahal", "Angkor Wat", "the Brandenburg Concertos",
 "Moby-Dick", "Hamlet", "the Divine Comedy", "the theory of evolution", "the steam engine", "insulin",
]
SCI = [
 "photosynthesis", "plate tectonics", "nuclear fission", "the water cycle", "natural selection",
 "electromagnetic induction", "the greenhouse effect", "cellular respiration", "the Doppler effect",
 "a black hole", "an algorithm", "compound interest", "a vaccine", "a semiconductor", "an ecosystem",
 "the immune system", "a recession", "a neural network", "magnetism", "erosion", "a democracy",
]
UNITS = [("seconds", "minutes", 60), ("minutes", "hours", 60), ("metres", "kilometres", 1000),
         ("millilitres", "litres", 1000), ("hours", "days", 24), ("days", "weeks", 7), ("months", "years", 12),
         ("inches", "feet", 12), ("pence", "pounds", 100), ("cents", "dollars", 100)]
TIMEWORDS = ["today", "now", "current", "latest", "recent", "modern", "present", "this year", "currently", "nowadays"]

def gen(rng: random.Random):
    R = []
    def add(q, t, n, kind):
        R.append({"query": q, "temporal_class": t, "needs_retrieval": n, "aug_kind": kind})
    # ---- current, implicit (no explicit cue) ----
    cur_org = ["Who serves as {o} {h}?", "Who is the {h} of {o}?", "Name the person who is {h} of {o}.",
               "Which executive holds the {h} job at {o}?", "{o} is led by which {h}?", "Who occupies the post of {h} at {o}?",
               "Who's in charge as {h} of {o}?", "Tell me who the {h} of {o} is."]
    for (h, o), tpl in itertools.product(ORG_OFFICE, cur_org):
        add(tpl.format(h=h, o=o), "current", 1, "current_implicit_org")
    cur_cty = ["Who is {c}'s {h}?", "Who is the {h} of {c}?", "Who holds the office of {h} in {c}?",
               "Name {c}'s {h}.", "{c} has which {h}?", "Who governs {c} as {h}?"]
    for (h, c), tpl in itertools.product(COUNTRY_OFFICE, cur_cty):
        add(tpl.format(h=h, c=c), "current", 1, "current_implicit_country")
    cur_met = ["What is the {m} of {p}?", "How large is the {m} of {p}?", "Give the {m} for {p}.",
               "What's {p}'s {m}?", "Roughly what is the {m} in {p}?"]
    for (m, p), tpl in itertools.product(METRIC_PLACE, cur_met):
        add(tpl.format(m=m, p=p), "current", 1, "current_implicit_metric")
    cur_title = ["Who holds {t} title?", "Who are the reigning champions of {t}?", "Which side is the defending champion of {t}?",
                 "Who is the title holder in {t}?", "Which team currently tops {t}?", "Who won {t} most recently?",
                 "Who is the champion of {t} right now?"]
    for t, tpl in itertools.product(TITLES, cur_title):
        add(tpl.format(t=t), "current", 1, "current_title")
    # ---- current, explicit ----
    cur_exp = ["Who is the {h} of {o} as of {y}?", "Who is serving as {h} of {o} in {y}?", "In {y}, who is {o}'s {h}?"]
    for (h, o), tpl in itertools.product(rng.sample(ORG_OFFICE, 20), cur_exp):
        add(tpl.format(h=h, o=o, y=rng.choice([2025, 2026])), "current", 1, "current_explicit_year")
    # ---- historical, implicit ----
    hist_before = ["Who was {o} before {p}?", "Who served as {o} prior to {p}?", "Who preceded {p} as {o}?",
                   "Who came after {p} as {o}?", "Who succeeded {p} as {o}?", "Before {p}, who was {o}?"]
    for (o, p), tpl in itertools.product(PREDECESSORS, hist_before):
        add(tpl.format(o=o, p=p), "historical", 1, "historical_relative_person")
    hist_event = ["Who led {c} during {e}?", "Who was the {h} of {c} at the time of {e}?", "Who ruled {c} when {e} happened?",
                  "Which leader governed {c} at the start of {e}?"]
    for (h, c), e in itertools.product(rng.sample(COUNTRY_OFFICE, 18), ANCHOR_EVENTS):
        tpl = rng.choice(hist_event); add(tpl.format(h=h, c=c, e=e), "historical", 1, "historical_event_anchor")
    hist_first = ["Who won the first {f}?", "Who received the inaugural {f}?", "Who claimed the earliest {f}?",
                  "Who was the original winner of the {f}?", "Which country won the very first {f}?"]
    for f, tpl in itertools.product(FIRSTS, hist_first):
        add(tpl.format(f=f), "historical", 1, "historical_first")
    hist_misc = ["What was the population of {p} at the end of {e}?", "How was {p} affected by {e}?",
                 "What happened in {p} after {e}?"]
    for (m, p), e in itertools.product(rng.sample(METRIC_PLACE, 10), rng.sample(ANCHOR_EVENTS, 10)):
        add(rng.choice(hist_misc).format(p=p, e=e), "historical", 1, "historical_event_anchor")
    # ---- historical, explicit year (minimal pair with the current rows) ----
    hist_year = ["Who was the {h} of {o} in {y}?", "In {y}, who was {o}'s {h}?", "Who served as {h} of {o} back in {y}?"]
    for (h, o), tpl in itertools.product(ORG_OFFICE, hist_year):
        add(tpl.format(h=h, o=o, y=rng.randint(1950, 2019)), "historical", 1, "historical_year_pair")
    for (h, c) in COUNTRY_OFFICE:
        add(f"Who was {c}'s {h} in {rng.randint(1900, 2018)}?", "historical", 1, "historical_year_pair")
    for t in TITLES:
        add(f"Who won {t} in {rng.randint(1960, 2019)}?", "historical", 1, "historical_year_pair")
    # ---- timeless with year-looking numbers, self-contained (needs 0) ----
    for _ in range(900):
        a = rng.randint(1000, 2099); b = rng.randint(2, 999); k = rng.randint(2, 13)
        q = rng.choice([
            f"Is {a} an even number?", f"Is {a} divisible by {k}?", f"What is {a} minus {b}?",
            f"Compute {a} plus {b}.", f"What is {a} multiplied by {k}?", f"Divide {a} by {k}.",
            f"What is the sum of {a} and {b}?", f"What is the difference between {a} and {b}?",
            f"Round {a} to the nearest hundred.", f"What is double {a}?", f"Is {a} greater than {a + b}?",
            f"What is {a} squared?", f"Write {a} in Roman numerals.", f"What is {a} modulo {k}?"])
        add(q, "timeless", 0, "timeless_numeric_selfcontained")
    for _ in range(500):
        small, big, f = rng.choice(UNITS); v = rng.choice([rng.randint(2, 40), rng.randint(1000, 2099)])
        q = rng.choice([f"How many {small} are there in {v} {big}?", f"Convert {v} {big} into {small}.",
                        f"{v} {big} equals how many {small}?", f"How many {big} make {v * f} {small}?"])
        add(q, "timeless", 0, "timeless_unit_selfcontained")
    # ---- closed-book trivial (needs 0) ----
    for _ in range(600):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        q = rng.choice([f"What do you get if you add {a} to {b}?", f"Subtract {b} from {a}.", f"Multiply {a} by {b % 12 + 1}.",
                        f"What is {a} take away {b}?", f"Calculate {a} plus {b}.", f"What is the product of {a} and {b % 12 + 1}?",
                        f"Is {a} bigger than {b}?", f"What number comes after {a}?", f"Count from {a} to {a + 3}."])
        add(q, "timeless", 0, "closed_book_arithmetic")
    for q in ["How many days does a normal year have?", "How many letters are in the English alphabet?",
              "What is the opposite of up?", "What is the opposite of light?", "What is the plural of mouse?",
              "Spell 'house'.", "How many wheels does a bicycle have?", "How many minutes make up an hour and a half?",
              "What colour do you get by mixing blue and yellow?", "Is ice colder than steam?", "How many corners does a square have?",
              "What is the first letter of the alphabet?", "How many zeros are in one thousand?", "What is the past tense of run?",
              "Which is heavier, a kilogram of iron or a gram of iron?", "How many hours are there in two days?",
              "What day comes after Monday?", "What month comes before June?", "Is the number seven odd?",
              "How many fingers are on one human hand?", "What is the opposite of wet?", "What rhymes with cat?",
              "How many quarters make a whole?", "What is the plural of child?", "What shape has three sides?"]:
        add(q, "timeless", 0, "closed_book_trivia")
    # ---- timeless with time words / years, needs evidence (needs 1) ----
    tw = ["What does the word '{w}' mean?", "How is '{w}' used as an adverb in English?", "Is '{w}' a noun or an adjective?",
          "What part of speech is '{w}'?", "Give a synonym for '{w}'."]
    for w, tpl in itertools.product(TIMEWORDS, tw):
        add(tpl.format(w=w), "timeless", 1, "timeless_timeword_meta")
    for s in SCI:
        y = rng.randint(1500, 2024)
        for q in [f"Explain {s}.", f"How does {s} work?", f"What is meant by {s}?", f"Describe {s} in simple terms.",
                  f"How does {s} work, for instance in a {y} textbook?", f"Why is {s} important today?",
                  f"What is the modern understanding of {s}?", f"How would a scientist in {y} describe {s}?"]:
            add(q, "timeless", 1, "timeless_concept")
    for x, tpl in itertools.product(WORKS, STABLE_FACT):
        add(tpl.format(x=x), "timeless", 1, "timeless_stable_fact")
    for x in WORKS:
        add(f"Which is the most famous recent edition of {x}?", "current", 1, "current_cue_pair")
        add(f"When was {x} created?", "historical", 1, "historical_when_pair")
    return R
