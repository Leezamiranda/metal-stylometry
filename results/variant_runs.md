(.venv) letze@LeezaLaptop:~/projects/metal-stylometry$ python3 src/make_variants.py --tag Death
Band-specific words to mask (from training songs only):
  Cannibal Corpse        cadavers, faster, precision, oozing, butcher, nails, crotch, ghastly, hack, psychotic, spinal, altered, cunt, pus, rancid, food, tattered, concealed, sliced, cracking
  Deicide                bible, crucifix, thou, satan, invert, contradiction, conviction, unbelieved, crucifixion, shrine, nazarene, hymn, thy, book, worthless, absent, forgot, fable, decision, likes
  Immolation             rays, tempt, purge, soldier, menacing, flourish, taunting, designed, world, wicked, comfort, far, fury, shed, promise, miles, service, faceless, somber, rampant
  Incantation            twilight, vitality, divinity, godly, horns, somber, sacrament, mortality, smite, luminous, evident, reside, delight, radiant, blasphemy, foul, thy, ethereal, vessel, obliteration
  Krisiun                timeless, holocaust, infamous, nations, storms, mundane, sorrowful, tyrant, entangled, infestation, redeemed, exalting, demonized, emplagued, pillage, slavery, pestilent, profanation, abyssal, blasting
  Malevolent Creation    begs, instilled, stroke, extinguish, sloth, dismantle, gun, gorging, entering, excitement, feeling, losing, begging, hard, loose, hit, tools, continue, tasting, life
  Napalm Death           access, norm, scab, trauma, angst, action, loathing, public, reject, goals, friends, duped, polluting, common, loose, plastic, box, holes, scale, crawl
  Six Feet Under         coffin, missing, gonna, motherfucker, holes, unburied, pus, murder, cutting, reaper, carving, slipping, torso, cut, skull, grave, fucking, suicide, neck, leaking
  Unleashed              thor, midgard, odin, battalions, welcome, yeah, northern, smiles, goodbye, courage, warriors, aim, brave, til, proud, wife, march, armies, slay, ride
  Vader                  jak, peter, names, genius, winged, mauser, chant, dla, spoken, fleshes, stars, myths, join, yell, lady, resting, shunned, dancing, ghosts, fever

Normalization removed 10,743 punctuation marks (1.2% of characters)
Masking replaced 14,733 of 161,056 words (9.1%)

--- lyrics_clean ---
Enter the crypts of hell to impregnate the dead Fresh burial sight, my hunger grows tonight Undying lust for cadaverous molestation Sights of death, building my sensations Ferocious need for decomposing bodies Sex with the dead, now I must breed Within the stiff corpse, planting my seed The taste of

--- lyrics_norm ---
Enter the crypts of hell to impregnate the dead Fresh burial sight my hunger grows tonight Undying lust for cadaverous molestation Sights of death building my sensations Ferocious need for decomposing bodies Sex with the dead now I must breed Within the stiff corpse planting my seed The taste of for

--- lyrics_masked ---
Enter the crypts of hell to impregnate the dead Fresh burial sight, my hunger grows tonight Undying lust for cadaverous molestation [MASK] of death, building my sensations Ferocious need for decomposing bodies [MASK] with the dead, now I must breed Within the stiff corpse, planting my seed The taste

--- lyrics_norm_masked ---
Enter the crypts of hell to impregnate the dead Fresh burial sight my hunger grows tonight Undying lust for cadaverous molestation [MASK] of death building my sensations Ferocious need for decomposing bodies [MASK] with the dead now I must breed Within the stiff corpse planting my seed The taste of

Added lyrics_norm, lyrics_masked, lyrics_norm_masked to data/processed/dataset_Death_cap100.parquet
(.venv) letze@LeezaLaptop:~/projects/metal-stylometry$ python3 src/make_variants.py --tag Power
Band-specific words to mask (from training songs only):
  Blind Guardian         imaginations, sadly, crow, ninth, determination, grail, praise, blindness, secretly, bears, deliverance, border, elves, hill, wheel, bind, watching, slaves, artificial, redeem
  Bloodbound             slipping, giants, north, dragon, creatures, arrows, dragons, hills, metal, northern, stake, oh, realm, olsson, creeping, corner, unholy, fly, dark, away
  Elvenking              pagan, runes, oak, branches, verse, glances, oaks, chant, thirteen, pains, reader, jarpen, clad, homes, shores, carried, dreadful, hushing, sudden, cloak
  Freedom Call           woah, metal, defenders, hail, warriors, gliding, oasis, deliverance, obsession, tall, creation, courageous, rock, fantasy, closing, oh, blackened, welcome, heavenly, emptiness
  Grave Digger           laughing, beggars, wife, scythe, pope, wheels, knights, victims, revelation, blood, die, life, hate, fighting, dancing, night, hell, attack, waking, stays
  HammerFall             templars, hammer, bang, hail, allied, repression, fury, breed, metal, immortalized, justice, united, rejected, solid, fighting, chime, vengeance, deal, warriors, charge
  Helloween              loving, metal, liar, woman, cat, heavy, spit, hey, blue, know, listen, time, short, laugh, wanna, crap, solo, crazy, kill, tv
  Iced Earth             empires, minions, crucify, sixth, traitors, cleanse, trials, falter, kingdoms, humanity, manipulate, blackness, manipulation, focused, awaits, vile, hypocrisy, pawns, tall, creed
  Kamelot                underneath, separate, await, velvet, rose, childhood, fragile, reunite, sunlight, careful, heartbeat, melody, like, embrace, pure, song, come, nights, life, day
  Rage                   lifes, arrogance, bout, profit, stuff, perception, rich, slave, fact, time, work, way, justice, company, bite, machine, straight, life, let, world

Normalization removed 16,642 punctuation marks (1.4% of characters)
Masking replaced 24,226 of 237,313 words (10.2%)

--- lyrics_clean ---
Hallelujah Hallelujah Hallelujah I am a preacher with a message for my people Over the world, scratching on the ground Looking for the peace that nobody has found I am a spokesman for a better way of living Love is the word and it can be heard If you are young the message can be sung Let me hear you

--- lyrics_norm ---
Hallelujah Hallelujah Hallelujah I am a preacher with a message for my people Over the world scratching on the ground Looking for the peace that nobody has found I am a spokesman for a better way of living Love is the word and it can be heard If you are young the message can be sung Let me hear you

--- lyrics_masked ---
[MASK] [MASK] [MASK] I am a preacher with a message for my people Over the [MASK], scratching on the ground Looking for the peace that nobody has found I am a spokesman for a better [MASK] of living Love is the word and it can be heard If you are young the message can be sung [MASK] me hear you sing

--- lyrics_norm_masked ---
[MASK] [MASK] [MASK] I am a preacher with a message for my people Over the [MASK] scratching on the ground Looking for the peace that nobody has found I am a spokesman for a better [MASK] of living Love is the word and it can be heard If you are young the message can be sung [MASK] me hear you sing

Added lyrics_norm, lyrics_masked, lyrics_norm_masked to data/processed/dataset_Power_cap100.parquet
(.venv) letze@LeezaLaptop:~/projects/metal-stylometry$ python3 src/make_variants.py --tag Mixed
Band-specific words to mask (from training songs only):
  Accept                 wild, metal, collision, pack, nights, train, heed, load, loaded, bite, brains, crimes, chains, midnight, carry, like, nature, ya, man, message
  Black Label Society    mourned, mend, remorse, bend, oh, shall, jesus, longer, reside, confide, visions, undying, yeah, rolling, salvation, rain, begins, fine, belief, gathering
  GWAR                   apes, admit, ohh, gwar, filthy, womb, wretched, bloated, dick, meat, slay, maggot, monster, fuck, babies, raped, kick, bloody, enslave, worm
  Iron Maiden            stranger, asking, eagle, sunrise, drift, divide, oh, soldier, tide, sunlight, barren, sails, rose, tribes, land, brave, afraid, riding, heaven, spirits
  King Diamond           abigail, sarah, voodoo, salem, spiders, mansion, loa, harry, miriam, puppet, malone, hallway, cellar, stairs, mommy, crypt, jonathan, jars, wheelchair, madame
  Korn                   deceiving, fuck, constantly, pounding, fucked, shit, hating, beg, cheating, okay, fine, feelings, pretend, fucking, ass, throwing, away, happy, talk, time
  Megadeth               caving, confess, train, clawing, family, revolution, lucky, lied, sleep, war, guns, enemy, brains, fatal, plans, order, lest, oh, secret, metal
  Sevendust              guilty, beautiful, picture, damage, shit, throw, create, know, feel, wash, away, gift, erased, come, life, stuck, beg, wipe, pain, like
  Venom                  satanic, deceased, summon, legions, mayhem, metal, steel, christ, ye, drug, ancient, virgin, nigh, hell, death, fallen, blessed, awaiting, ripping, skies
  W.A.S.P.               idol, superman, charlie, morgue, crimson, mama, jesse, wild, doctor, orphan, roamed, ya, neon, boy, flame, thunder, oh, messiah, disciples, ooh

Normalization removed 24,976 punctuation marks (2.2% of characters)
Masking replaced 23,175 of 226,863 words (10.2%)

--- lyrics_clean ---
Nice try - if you wanna get me down You gotta shoot me twice Watch out - you better move aside And try to realize - that I'm alive Why me - can't you see You're nothing I wanna be Back off - you should be aware Diggin' in the dirt - isn't fair What do you want from me What are you trying to say Stop

--- lyrics_norm ---
Nice try if you wanna get me down You gotta shoot me twice Watch out you better move aside And try to realize that I'm alive Why me can't you see You're nothing I wanna be Back off you should be aware Diggin in the dirt isn't fair What do you want from me What are you trying to say Stop diggin in th

--- lyrics_masked ---
Nice try - if you wanna get me down You gotta shoot me twice Watch out - you better move aside And try to realize - that I'm alive Why me - can't you see You're nothing I wanna be Back off - you should be aware [MASK]' in the dirt - isn't fair What do you want from me What are you trying to say Stop

--- lyrics_norm_masked ---
Nice try if you wanna get me down You gotta shoot me twice Watch out you better move aside And try to realize that I'm alive Why me can't you see You're nothing I wanna be Back off you should be aware [MASK] in the dirt isn't fair What do you want from me What are you trying to say Stop diggin in th

Added lyrics_norm, lyrics_masked, lyrics_norm_masked to data/processed/dataset_Mixed_cap100.parquet