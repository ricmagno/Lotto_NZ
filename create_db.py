import sqlite3

conn = sqlite3.connect('draw_predictions.db')
c = conn.cursor()

c.execute("CREATE TABLE bullseye (Draw int,  Ball_1 real, Ball_2 real, Ball_3 real, Ball_4 real, Ball_5 real, Ball_6 real, Number int, Sorted binary, Multivariable binary, Version tinyint)")

c.execute("CREATE TABLE lotto (Draw int,  Ball_1 real, Ball_2 real, Ball_3 real, Ball_4 real, Ball_5 real, Ball_6 real, Bonus real, Powerball real, Sorted binary, Multivariable binary, Version tinyint)")




# c.execute("CREATE TABLE vehicle (frame int, score real)")
