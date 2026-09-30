"""States, capitals, cities, templates for EditHunt.

City lists are hand-curated: no capitals, no names containing a state name,
no well-known multi-state names (see AMBIGUOUS). The validity filter
(experiments/phase0_validity.py) decides which (city, template) instances the
model actually gets right.
"""
from __future__ import annotations

CAPITALS = {
    "Alabama": "Montgomery", "Alaska": "Juneau", "Arizona": "Phoenix",
    "Arkansas": "Little Rock", "California": "Sacramento", "Colorado": "Denver",
    "Connecticut": "Hartford", "Delaware": "Dover", "Florida": "Tallahassee",
    "Georgia": "Atlanta", "Hawaii": "Honolulu", "Idaho": "Boise",
    "Illinois": "Springfield", "Indiana": "Indianapolis", "Iowa": "Des Moines",
    "Kansas": "Topeka", "Kentucky": "Frankfort", "Louisiana": "Baton Rouge",
    "Maine": "Augusta", "Maryland": "Annapolis", "Massachusetts": "Boston",
    "Michigan": "Lansing", "Minnesota": "Saint Paul", "Mississippi": "Jackson",
    "Missouri": "Jefferson City", "Montana": "Helena", "Nebraska": "Lincoln",
    "Nevada": "Carson City", "New Hampshire": "Concord", "New Jersey": "Trenton",
    "New Mexico": "Santa Fe", "New York": "Albany", "North Carolina": "Raleigh",
    "North Dakota": "Bismarck", "Ohio": "Columbus", "Oklahoma": "Oklahoma City",
    "Oregon": "Salem", "Pennsylvania": "Harrisburg", "Rhode Island": "Providence",
    "South Carolina": "Columbia", "South Dakota": "Pierre", "Tennessee": "Nashville",
    "Texas": "Austin", "Utah": "Salt Lake City", "Vermont": "Montpelier",
    "Virginia": "Richmond", "Washington": "Olympia", "West Virginia": "Charleston",
    "Wisconsin": "Madison", "Wyoming": "Cheyenne",
}
STATES = list(CAPITALS)
assert len(STATES) == 50

# Names that exist prominently in several states; never used as subject cities.
AMBIGUOUS = {
    "Portland", "Springfield", "Columbus", "Arlington", "Aurora", "Jacksonville",
    "Kansas City", "Richmond", "Glendale", "Pasadena", "Albany", "Charleston",
    "Columbia", "Rochester", "Bloomington", "Lexington", "Wilmington", "Newark",
    "Manchester", "Burlington", "Fayetteville", "Greenville", "Athens", "Salem",
    "Plymouth", "Auburn", "Newport", "Lafayette", "Oxford", "Manhattan",
    "Jackson", "Lincoln", "Concord", "Augusta", "Dover", "Madison", "Frankfort",
    "Alexandria", "Texarkana", "Bristol", "Franklin", "Clinton", "Kingston",
    "Monroe", "Cambridge", "Fairfield", "Hamilton", "Troy", "Vienna", "Paris",
    "Berlin", "Florence", "Rome", "Warwick", "Huntington", "Amherst", "Hanover",
    "Bowling Green", "Portsmouth", "Durham",
    "Decatur", "Carmel", "Moscow", "Quincy", "Lancaster", "Canton", "Aberdeen",
    "Milford", "Middletown", "Smyrna", "Conway", "Meridian", "Waterloo",
    "Hillsboro", "Carlsbad", "Farmington", "Covington", "Beaufort",
    "Independence", "Sheridan", "Rutland", "Brunswick", "Elizabeth", "Rogers",
}

CITIES = {
    "Alabama": ["Birmingham", "Huntsville", "Mobile", "Tuscaloosa", "Dothan", "Gadsden", "Decatur", "Hoover"],
    "Alaska": ["Anchorage", "Fairbanks", "Ketchikan", "Sitka", "Nome", "Kodiak", "Wasilla", "Homer"],
    "Arizona": ["Tucson", "Scottsdale", "Flagstaff", "Tempe", "Mesa", "Sedona", "Yuma", "Chandler"],
    "Arkansas": ["Hot Springs", "Fort Smith", "Bentonville", "Jonesboro", "Pine Bluff", "Conway", "Rogers"],
    "California": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Fresno", "Oakland", "Bakersfield", "Anaheim", "Santa Barbara", "Palo Alto", "Berkeley", "Malibu"],
    "Colorado": ["Boulder", "Aspen", "Fort Collins", "Pueblo", "Vail", "Durango", "Telluride", "Breckenridge"],
    "Connecticut": ["New Haven", "Stamford", "Bridgeport", "Greenwich", "Waterbury", "Norwalk", "Danbury", "Mystic"],
    "Delaware": ["Rehoboth Beach", "Lewes", "Smyrna", "Middletown", "Milford", "Seaford"],
    "Florida": ["Miami", "Orlando", "Tampa", "Fort Lauderdale", "Key West", "Gainesville", "Sarasota", "Naples", "Pensacola", "Daytona Beach"],
    "Georgia": ["Savannah", "Macon", "Valdosta", "Marietta", "Alpharetta", "Dahlonega", "Statesboro"],
    "Hawaii": ["Hilo", "Lahaina", "Kailua", "Kona", "Waikiki", "Lihue"],
    "Idaho": ["Pocatello", "Twin Falls", "Coeur d'Alene", "Sun Valley", "Nampa", "Moscow", "Ketchum"],
    "Illinois": ["Chicago", "Peoria", "Naperville", "Rockford", "Joliet", "Evanston", "Champaign", "Urbana"],
    "Indiana": ["Fort Wayne", "South Bend", "Gary", "Evansville", "Terre Haute", "Muncie", "Carmel", "West Lafayette"],
    "Iowa": ["Cedar Rapids", "Davenport", "Dubuque", "Sioux City", "Ames", "Council Bluffs", "Waterloo"],
    "Kansas": ["Wichita", "Lawrence", "Dodge City", "Overland Park", "Olathe", "Salina", "Hutchinson"],
    "Kentucky": ["Louisville", "Paducah", "Owensboro", "Covington", "Berea", "Elizabethtown", "Hopkinsville"],
    "Louisiana": ["New Orleans", "Shreveport", "Lake Charles", "Houma", "Natchitoches", "Thibodaux", "Ruston"],
    "Maine": ["Bangor", "Bar Harbor", "Lewiston", "Kennebunkport", "Biddeford", "Waterville", "Ogunquit"],
    "Maryland": ["Baltimore", "Bethesda", "Rockville", "Frederick", "Silver Spring", "Ocean City", "Hagerstown", "Gaithersburg"],
    "Massachusetts": ["Worcester", "Lowell", "Nantucket", "Martha's Vineyard", "Provincetown", "Pittsfield", "Brockton", "Quincy"],
    "Michigan": ["Detroit", "Grand Rapids", "Ann Arbor", "Flint", "Kalamazoo", "Dearborn", "Traverse City", "Saginaw"],
    "Minnesota": ["Minneapolis", "Duluth", "St. Cloud", "Mankato", "Eden Prairie", "Moorhead", "Bemidji"],
    "Mississippi": ["Biloxi", "Gulfport", "Tupelo", "Hattiesburg", "Natchez", "Vicksburg", "Meridian"],
    "Missouri": ["St. Louis", "Branson", "Joplin", "Hannibal", "Independence", "Cape Girardeau", "St. Joseph"],
    "Montana": ["Billings", "Missoula", "Bozeman", "Great Falls", "Butte", "Kalispell", "Whitefish"],
    "Nebraska": ["Omaha", "Grand Island", "Kearney", "Scottsbluff", "North Platte"],
    "Nevada": ["Las Vegas", "Reno", "Henderson", "Elko", "Lake Tahoe", "Sparks", "Laughlin"],
    "New Hampshire": ["Nashua", "Keene", "Laconia", "Derry", "Hampton Beach"],
    "New Jersey": ["Jersey City", "Hoboken", "Atlantic City", "Princeton", "Camden", "Paterson", "Asbury Park", "Elizabeth"],
    "New Mexico": ["Albuquerque", "Las Cruces", "Roswell", "Taos", "Los Alamos", "Farmington", "Carlsbad"],
    "New York": ["Buffalo", "Syracuse", "Yonkers", "Ithaca", "Poughkeepsie", "Brooklyn", "Utica", "Schenectady"],
    "North Carolina": ["Charlotte", "Greensboro", "Asheville", "Chapel Hill", "Winston-Salem", "Boone", "Cary", "Outer Banks"],
    "North Dakota": ["Fargo", "Grand Forks", "Minot", "Williston", "Dickinson"],
    "Ohio": ["Cleveland", "Cincinnati", "Toledo", "Akron", "Dayton", "Youngstown", "Sandusky", "Canton"],
    "Oklahoma": ["Tulsa", "Norman", "Stillwater", "Lawton", "Broken Arrow", "Muskogee", "Edmond"],
    "Oregon": ["Eugene", "Bend", "Medford", "Corvallis", "Beaverton", "Ashland", "Hillsboro", "Astoria"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Scranton", "Erie", "Gettysburg", "Hershey", "Lancaster"],
    "Rhode Island": ["Pawtucket", "Cranston", "Woonsocket", "Narragansett", "Westerly"],
    "South Carolina": ["Myrtle Beach", "Hilton Head", "Spartanburg", "Rock Hill", "Beaufort", "Summerville"],
    "South Dakota": ["Sioux Falls", "Rapid City", "Deadwood", "Sturgis", "Brookings", "Aberdeen"],
    "Tennessee": ["Memphis", "Knoxville", "Chattanooga", "Gatlinburg", "Murfreesboro", "Clarksville", "Pigeon Forge"],
    "Texas": ["Houston", "Dallas", "San Antonio", "El Paso", "Fort Worth", "Lubbock", "Corpus Christi", "Waco", "Amarillo", "Galveston", "Plano", "Laredo"],
    "Utah": ["Provo", "Ogden", "Moab", "Park City", "St. George", "Orem", "Logan"],
    "Vermont": ["Stowe", "Brattleboro", "Bennington", "Rutland", "Middlebury", "Killington"],
    "Virginia": ["Norfolk", "Roanoke", "Charlottesville", "Williamsburg", "Newport News", "Blacksburg", "Chesapeake", "Lynchburg"],
    "Washington": ["Seattle", "Spokane", "Tacoma", "Bellingham", "Yakima", "Walla Walla", "Redmond", "Everett"],
    "West Virginia": ["Morgantown", "Wheeling", "Parkersburg", "Beckley", "Harpers Ferry"],
    "Wisconsin": ["Milwaukee", "Green Bay", "Kenosha", "Oshkosh", "Eau Claire", "La Crosse", "Racine", "Appleton"],
    "Wyoming": ["Casper", "Laramie", "Jackson Hole", "Cody", "Sheridan", "Rock Springs"],
}


def _clean() -> None:
    for s, cs in CITIES.items():
        seen, out = set(), []
        for c in cs:
            if c in seen or c in AMBIGUOUS or c in CAPITALS.values():
                continue
            if any(st.lower() in c.lower() for st in STATES):
                continue
            seen.add(c)
            out.append(c)
        CITIES[s] = out


_clean()
CITY2STATE = {c: s for s, cs in CITIES.items() for c in cs}
assert len(CITY2STATE) == sum(len(v) for v in CITIES.values()), "city in two states"

# ---------------------------------------------------------------- templates
# Few-shot demo states are reserved: never source/target in the same run.
FS_A = [("Chicago", "Illinois"), ("Miami", "Florida"), ("Seattle", "Washington")]
FS_B = [("Buffalo", "New York"), ("Tucson", "Arizona"), ("Detroit", "Michigan")]
RESERVED_STATES = {s for _, s in FS_A + FS_B}


def _fs(demos, fmt):
    return " ".join(fmt.format(city=c) + " " + CAPITALS[s] + "." for c, s in demos) + " " + fmt


_ZS1 = "The state containing {city} has its capital in"
_ZS2 = "The capital of the state where {city} is located is"
_HO = "{city} is in a state whose capital is"

TEMPLATES = {
    "zs1": _ZS1,
    "zs2": _ZS2,
    "fs1": _fs(FS_A, _ZS1),
    "fs2": _fs(FS_B, _ZS2),
    # held out: never used to build interventions
    "ho_fs": _fs(FS_A, _HO),
    "ho_zs": "Q: What is the capital of the state that {city} is in?\nA:",
}
TRAIN_TEMPLATES = ["fs1", "fs2", "zs1", "zs2"]
HELDOUT_TEMPLATES = ["ho_fs", "ho_zs"]

# state-belief (hop-1 readout)
STATE_Q = ("Q: Which US state is Chicago in?\nA: The state of Illinois.\n"
           "Q: Which US state is Miami in?\nA: The state of Florida.\n"
           "Q: Which US state is {city} in?\nA: The state of")
# held-out state-belief wording: never used to train keep-penalties (Phase 3 overfitting control)
STATE_Q_HO = ("Seattle is located in the state of Washington.\nMiami is located in the state of Florida.\n"
              "{city} is located in the state of")
# Country probe (collateral-damage check). The zero-shot form "{city} is a city in the country of" is NOT
# state-independent: base Qwen2.5-1.5B answers " Texas" etc. (the state) with p~0.3-0.7, so full-vocab KL on it
# measures the intended state change, not collateral damage (it was ~1.0 nats for working edits). We use a
# few-shot prompt and read only a restricted set of country answers: P(US) = mass of the first N_US
# candidates among COUNTRY_CANDS (exact multi-token scores, renormalised). Damage = change in P(US).
COUNTRY_Q_ZS = "{city} is a city in the country of"
COUNTRY_Q = ("Lyon is a city in the country of France.\nOsaka is a city in the country of Japan.\n"
             "Toronto is a city in the country of Canada.\n{city} is a city in the country of")
COUNTRY_CANDS = ["the United States", "United States", "America", "the USA", "Canada", "Mexico", "France",
                 "Japan", "England", "the United Kingdom", "Germany", "Spain", "Italy", "China", "India",
                 "Australia"]
N_US = 4

GENERIC_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old barn.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She poured a cup of coffee and opened the morning newspaper.",
    "The committee will review the proposal at next week's meeting.",
    "Prime numbers have exactly two distinct positive divisors.",
    "He tied his shoes, grabbed his keys, and walked out the door.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Mount Everest is the highest mountain above sea level.",
    "The software update fixed several bugs in the login system.",
    "Children played soccer in the park until the sun went down.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The novel follows a young detective solving her first case.",
    "Our flight was delayed by three hours because of the storm.",
    "The orchestra tuned their instruments before the concert began.",
    "Regular exercise improves both physical and mental health.",
    "The museum's new exhibit features ancient Egyptian artifacts.",
    "A balanced diet includes fruits, vegetables, and whole grains.",
    "The printer ran out of ink in the middle of the report.",
    "Bees communicate the location of flowers through a waggle dance.",
    "The train leaves the station every morning at seven o'clock.",
    "Quantum mechanics describes the behavior of very small particles.",
    "The cat curled up on the windowsill and fell asleep.",
    "Investors reacted cautiously to the latest inflation figures.",
    "The garden was full of tomatoes, peppers, and fresh basil.",
    "Shakespeare wrote many plays that are still performed today.",
    "The students gathered in the library to study for exams.",
    "A new bridge will connect the two sides of the river.",
    "The chef sharpened his knives before the dinner service.",
    "Electric cars are becoming more common on highways.",
    "The dog barked loudly when the mail carrier arrived.",
    "Mathematics is the language in which the universe is written.",
    "The hikers reached the summit just before sunrise.",
    "Heavy rain caused flooding in several low-lying areas.",
    "The teacher explained the lesson using colorful diagrams.",
    "Most mammals give birth to live young rather than laying eggs.",
    "The company announced record profits for the third quarter.",
    "Her grandmother knitted a warm scarf for the winter.",
    "The telescope revealed thousands of distant galaxies.",
    "He practiced the piano for two hours every evening.",
    "The ancient castle stood on a hill overlooking the valley.",
    "Vaccines train the immune system to recognize pathogens.",
    "The bakery sells fresh bread and pastries every morning.",
    "A strong password contains letters, numbers, and symbols.",
    "The river winds through the forest toward the sea.",
    "They painted the living room a pale shade of green.",
    "The spacecraft entered orbit after a six month journey.",
    "Autumn leaves turned red and gold along the country road.",
    "The lawyer presented new evidence to the jury.",
    "Octopuses are known for their intelligence and camouflage.",
    "The band released their first album last spring.",
]
assert len(GENERIC_SENTENCES) == 50


def study_states(min_cities: int = 6) -> list[str]:
    return [s for s in STATES if s not in RESERVED_STATES and len(CITIES[s]) >= min_cities]


if __name__ == "__main__":
    n = sum(len(v) for v in CITIES.values())
    print(f"{n} cities over {len(CITIES)} states; study states (>=6 cities): {len(study_states())}")
    for k, t in TEMPLATES.items():
        print(k, repr(t.format(city="Dallas")))


# ---------------------------------------------------------------- hidden state-level readouts (task T3)
# Never exposed to agent tools; used to check that an edit moved the *state variable*, not one answer.
ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
NICKNAMES = {
    "Alabama": "Yellowhammer State", "Alaska": "Last Frontier", "Arizona": "Grand Canyon State",
    "Arkansas": "Natural State", "California": "Golden State", "Colorado": "Centennial State",
    "Connecticut": "Constitution State", "Delaware": "First State", "Florida": "Sunshine State",
    "Georgia": "Peach State", "Hawaii": "Aloha State", "Idaho": "Gem State", "Illinois": "Prairie State",
    "Indiana": "Hoosier State", "Iowa": "Hawkeye State", "Kansas": "Sunflower State",
    "Kentucky": "Bluegrass State", "Louisiana": "Pelican State", "Maine": "Pine Tree State",
    "Maryland": "Old Line State", "Massachusetts": "Bay State", "Michigan": "Great Lakes State",
    "Minnesota": "North Star State", "Mississippi": "Magnolia State", "Missouri": "Show-Me State",
    "Montana": "Treasure State", "Nebraska": "Cornhusker State", "Nevada": "Silver State",
    "New Hampshire": "Granite State", "New Jersey": "Garden State", "New Mexico": "Land of Enchantment",
    "New York": "Empire State", "North Carolina": "Tar Heel State", "North Dakota": "Peace Garden State",
    "Ohio": "Buckeye State", "Oklahoma": "Sooner State", "Oregon": "Beaver State",
    "Pennsylvania": "Keystone State", "Rhode Island": "Ocean State", "South Carolina": "Palmetto State",
    "South Dakota": "Mount Rushmore State", "Tennessee": "Volunteer State", "Texas": "Lone Star State",
    "Utah": "Beehive State", "Vermont": "Green Mountain State", "Virginia": "Old Dominion",
    "Washington": "Evergreen State", "West Virginia": "Mountain State", "Wisconsin": "Badger State",
    "Wyoming": "Equality State",
}
# key -> (template, state -> answer string)
READOUTS = {
    "abbr": ("Chicago, IL\nMiami, FL\nSeattle, WA\n{city},", ABBR),
    "abbr_addr": ("Ship to: 12 Oak Ave, Chicago, IL 60601\nShip to: 400 Pine St, Miami, FL 33101\n"
                  "Ship to: 9 Elm Rd, {city},", ABBR),
    "nick": ("Chicago is in the state nicknamed the Prairie State. Miami is in the state nicknamed the Sunshine "
             "State. {city} is in the state nicknamed the", NICKNAMES),
}
