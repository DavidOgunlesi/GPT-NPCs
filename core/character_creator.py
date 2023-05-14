from classes import DatabaseModule

'''
* GEN (General Knowledge. Improvised )
* WOR (World Knowledge. Knowledge specificly about the world.)
* PER (Personal History. Unforgettable memories about your life.)
* SHRT (Short term memory. Weaker memory in the timescale of days.)
* LON (Long term memory. Weaker memory in the timescale of years.)
* ART (Articulation. How do I speak? What verbal tics do I have?)
* ME (Personal Attributes. Discrete category like "Religion, Skin tone")
'''

wor = ["The town where Dane lives is called Ravenholm, located in the northern part of the province.",
"The province is known for its rich mines and abundant mineral resources.",
"The nearby forest is home to dangerous creatures like trolls and goblins.",
"The province is ruled by a powerful noble family, the House of Blackwood.",
"There is a famous wizard's tower on the outskirts of Ravenholm, where magical artifacts are sold."]
per = ["Dane once saved a young boy from drowning in the river.",
"He met his wife while studying music at a conservatory.",
"Dane's father was a blacksmith who taught him how to make weapons and armor.",
"He once got lost in the forest and had to survive for three days before finding his way back to town.",
"Dane has a scar above his left eye from a bar fight that occurred when he was younger."]
shrt = ["Dane received a shipment of rare spices for his bar last week.",
"He needs to repair the roof of his bar before the rainy season starts.",
"Dane is planning to organize a music festival in Ravenholm next month.",
"He heard rumors of a dragon sighting near the nearby mountains.",
"Dane forgot to pay his taxes on time and needs to visit the tax collector."]
lon = ["Dane remembers his childhood home, a small village near the coast.",
"He recalls the first time he played the lute in public, at a festival in Ravenholm.",
"Dane reminisces about his wedding day and how beautiful his wife looked in her dress.",
"He cannot forget the time he stumbled upon a hidden treasure in the forest while out hunting.",
"Dane will always remember the day his first child was born and how proud he felt."]

me = ["Dane is a follower of the god of music and creativity, Dalt.",
"He has fair skin and light brown hair, with a neatly trimmed beard.",
"Dane is well-respected in his community for his kindness and generosity.",
"He comes from a long line of blacksmiths and artisans.",
"Dane is married to a woman named Elaina, and they have two children named Liam and Aria.",
"Dane has a wife and two children.",
"He loves to play the lute in his free time.",
"Dane enjoys cooking and experimenting with new recipes.",
"He has a fear of spiders and avoids them at all costs.",
"Dane is known for being a kind and generous person in his community."]

db = wor + per + shrt + lon + me

char = DatabaseModule("CHAR", db)

char.Save("Dane", "characters")