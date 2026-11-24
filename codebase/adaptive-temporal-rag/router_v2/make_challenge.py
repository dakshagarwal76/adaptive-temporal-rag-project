"""Hand-written v2 challenge set (written before any v2 training data was generated).
Labels follow the v2 schema: temporal in {current, historical, timeless};
needs_retrieval = 0 only for self-contained questions (arithmetic, unit/calendar facts,
word-level trivia) that need no evidence. Everything else needs retrieval."""
import json
C = []
def add(cat, t, n, qs):
    for q in qs: C.append({"category": cat, "query": q, "gold_temporal": t, "gold_needs": n})
add("CA_timeless_numeric_or_timeword_selfcontained", "timeless", 0, [
 "What is 1850 divided by 2?", "Is 1987 an odd number?", "What do you get when you subtract 1914 from 2000?",
 "How many weeks are in 364 days?", "Is 2031 divisible by 3?", "Add 1776 and 224.",
 "How many centimetres are in 1905 millimetres?", "What is half of 2012?",
 "How many grams are in 3 kilograms?", "What is the remainder when 1999 is divided by 10?"])
add("CA2_timeless_timeword_needs_evidence", "timeless", 1, [
 "Why do we have a leap second occasionally added to clocks?",
 "What does the adverb 'nowadays' mean?", "In chemistry, what is a current of electrons called?",
 "What is the latest stage of mitosis called?", "Which is the deepest ocean trench on Earth?",
 "How does radiocarbon dating estimate the age of bones from 3000 BC?",
 "What is the boiling point of water at sea level in 2024 units of Celsius?",
 "What does the prefix 'pre-' mean in the word 'prehistoric'?",
 "Which gas makes up most of the atmosphere today?", "What is the modern scientific name for the domestic dog?"])
add("CB_current_implicit", "current", 1, [
 "Who runs the European Space Agency?", "Who is the head of the World Health Organization?",
 "What is the unemployment rate in Canada?", "Who chairs the board of Samsung Electronics?",
 "Who is Australia's governor-general?", "What is the exchange rate of the euro to the yen?",
 "Who manages Real Madrid's first team?", "Which club tops the Bundesliga table?",
 "Who is the reigning world chess champion?", "What is the population of Nigeria?",
 "Who is the mayor of Paris?", "Who heads the International Monetary Fund?"])
add("CC_historical_implicit", "historical", 1, [
 "Who led the Soviet Union before Gorbachev?", "Which country won the very first Eurovision Song Contest?",
 "Who was Microsoft's chief executive before Satya Nadella?", "Who was the US president when the Apollo 11 mission landed?",
 "Who won the first ever Tour de France?", "What was the capital of Brazil before Brasilia?",
 "Who was the British monarch during the Battle of Waterloo?", "Who ruled Egypt when the Suez Canal was nationalised?",
 "Which nation hosted the Olympics immediately before the London Games of the twenty-first century?",
 "Who directed Disney's first full-length animated feature?",
 "Who preceded Angela Merkel as German chancellor?", "Who was Pope before Benedict XVI?"])
add("CD_year_anchored", "historical", 1, [
 "Who won the Ballon d'Or in 2008?", "Which film won the Palme d'Or in 1994?",
 "Who was the Canadian prime minister in 1985?", "Who won the Booker Prize in 2002?",
 "Which team won the Stanley Cup in 1999?", "Who was Secretary-General of NATO in 2012?"])
add("CD2_recent_year_current", "current", 1, [
 "Who is expected to host the 2027 Rugby World Cup?", "Which city will host the 2030 Winter Olympics?"])
add("CE_closed_book_trivial", "timeless", 0, [
 "What is nine multiplied by six?", "How many seconds are in a minute?", "What is the opposite of tall?",
 "How many months are in a year?", "What is 15 minus 6?", "How many legs does a spider have?",
 "Spell the word 'cat' backwards.", "What is 100 divided by 4?", "How many sides does a triangle have?",
 "Is fire hot or cold?"])
add("CF_timeless_factual", "timeless", 1, [
 "Who wrote the novel Middlemarch?", "What is the chemical symbol for tungsten?",
 "Define the term photosphere.", "Explain what a binary search tree is.", "What is the function of the pancreas?",
 "Which composer wrote The Rite of Spring?", "What is the atomic number of neon?",
 "Give a brief definition of inflation in economics.", "What is a prime number?", "Who painted The Night Watch?"])
with open("challenge_v2.jsonl", "w") as f:
    for i, r in enumerate(C):
        r["id"] = f"ch{i:03d}"; f.write(json.dumps(r) + "\n")
print(len(C))
