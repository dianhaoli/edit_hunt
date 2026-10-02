"""Private downstream readouts of a city's COUNTRY (lever A of AUDIT_T2V2 §8.3: grade "keep" on uses of the kept fact
that the agent never sees). Country -> capital / currency for the 64 countries of the city_pop survivors.
Surfaces: value -> alternative answer strings. Templates are few-shot with demo cities outside the kept set.
"""
from __future__ import annotations

CAPITAL = {
    "China": "Beijing", "United States": "Washington", "India": "New Delhi", "Brazil": "Brasilia", "Russia": "Moscow",
    "Japan": "Tokyo", "Indonesia": "Jakarta", "Mexico": "Mexico City", "Iran": "Tehran", "Germany": "Berlin",
    "Colombia": "Bogota", "Italy": "Rome", "Pakistan": "Islamabad", "Nigeria": "Abuja", "United Kingdom": "London",
    "Angola": "Luanda", "Turkey": "Ankara", "Canada": "Ottawa", "France": "Paris", "Zambia": "Lusaka", "Poland": "Warsaw",
    "Argentina": "Buenos Aires", "Mozambique": "Maputo", "Australia": "Canberra", "Tanzania": "Dodoma",
    "Ukraine": "Kyiv", "Venezuela": "Caracas", "Philippines": "Manila", "Ethiopia": "Addis Ababa",
    "Ivory Coast": "Yamoussoukro", "Cuba": "Havana", "Romania": "Bucharest", "Iraq": "Baghdad", "Bulgaria": "Sofia",
    "Ghana": "Accra", "Ecuador": "Quito", "Spain": "Madrid", "Sweden": "Stockholm", "Morocco": "Rabat", "Mali": "Bamako",
    "South Sudan": "Juba", "Norway": "Oslo", "Uzbekistan": "Tashkent", "Chile": "Santiago", "Sudan": "Khartoum",
    "Uruguay": "Montevideo", "Guinea": "Conakry", "Afghanistan": "Kabul", "Switzerland": "Bern", "Albania": "Tirana",
    "Portugal": "Lisbon", "Kazakhstan": "Astana", "Egypt": "Cairo", "Kenya": "Nairobi", "Azerbaijan": "Baku",
    "Tunisia": "Tunis", "Gabon": "Libreville", "Algeria": "Algiers", "New Zealand": "Wellington",
    "Central African Republic": "Bangui", "Malaysia": "Kuala Lumpur", "Greece": "Athens", "Cameroon": "Yaounde",
    "Myanmar": "Naypyidaw"}
CAPITAL_ALIASES = {"Washington": ["Washington, D.C.", "Washington DC"], "Brasilia": ["Brasília"], "Bogota": ["Bogotá"],
                   "Kyiv": ["Kiev"], "Yamoussoukro": ["Abidjan"], "Dodoma": ["Dar es Salaam"], "Astana": ["Nur-Sultan"],
                   "Yaounde": ["Yaoundé"], "Naypyidaw": ["Naypyitaw", "Yangon"], "New Delhi": ["Delhi"]}

CURRENCY = {
    "China": "yuan", "United States": "dollar", "India": "rupee", "Brazil": "real", "Russia": "ruble", "Japan": "yen",
    "Indonesia": "rupiah", "Mexico": "peso", "Iran": "rial", "Germany": "euro", "Colombia": "peso", "Italy": "euro",
    "Pakistan": "rupee", "Nigeria": "naira", "United Kingdom": "pound", "Angola": "kwanza", "Turkey": "lira",
    "Canada": "dollar", "France": "euro", "Zambia": "kwacha", "Poland": "zloty", "Argentina": "peso",
    "Mozambique": "metical", "Australia": "dollar", "Tanzania": "shilling", "Ukraine": "hryvnia", "Venezuela": "bolivar",
    "Philippines": "peso", "Ethiopia": "birr", "Ivory Coast": "franc", "Cuba": "peso", "Romania": "leu", "Iraq": "dinar",
    "Bulgaria": "lev", "Ghana": "cedi", "Ecuador": "dollar", "Spain": "euro", "Sweden": "krona", "Morocco": "dirham",
    "Mali": "franc", "South Sudan": "pound", "Norway": "krone", "Uzbekistan": "som", "Chile": "peso", "Sudan": "pound",
    "Uruguay": "peso", "Guinea": "franc", "Afghanistan": "afghani", "Switzerland": "franc", "Albania": "lek",
    "Portugal": "euro", "Kazakhstan": "tenge", "Egypt": "pound", "Kenya": "shilling", "Azerbaijan": "manat",
    "Tunisia": "dinar", "Gabon": "franc", "Algeria": "dinar", "New Zealand": "dollar", "Central African Republic": "franc",
    "Malaysia": "ringgit", "Greece": "euro", "Cameroon": "franc", "Myanmar": "kyat"}
CURRENCY_ALIASES = {"yuan": ["renminbi", "Chinese yuan"], "ruble": ["rouble"], "real": ["Brazilian real"],
                    "zloty": ["złoty"], "krona": ["Swedish krona"], "krone": ["Norwegian krone"],
                    "bolivar": ["bolívar"], "rial": ["Iranian rial"]}

READOUTS = {
    "CountryCapital": ("Grenoble is in a country whose capital is Paris. Hiroshima is in a country whose capital is Tokyo. "
                       "Mendoza is in a country whose capital is Buenos Aires. %s is in a country whose capital is",
                       CAPITAL, CAPITAL_ALIASES),
    "Currency": ("In Grenoble people pay with the euro. In Hiroshima people pay with the yen. In Mendoza people pay with the peso. "
                 "In %s people pay with the", CURRENCY, CURRENCY_ALIASES),
}


def verbalizer(name: str) -> dict:
    """value -> surfaces, over the distinct values of the readout."""
    _, table, al = READOUTS[name]
    return {v: [v] + al.get(v, []) for v in sorted(set(table.values()))}
