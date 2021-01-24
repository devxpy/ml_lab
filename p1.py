import csv


hypothesis = ["0"] * 6
columns = 7

with open("p1.csv", "r") as csvfile:
    datareader = csv.reader(csvfile, delimiter=",")
    next(datareader)  # skip header

    for train_data in datareader:
        print("t:", train_data)

        if train_data[columns - 1] == "1":
            for y in range(0, columns - 1):
                if hypothesis[y] == train_data[y]:
                    pass
                elif hypothesis[y] == "0":
                    hypothesis[y] = train_data[y]
                else:
                    hypothesis[y] = "?"

        print("h:", hypothesis)

print("Maximally Specific set")
print("<", end=" ")
for i in range(0, len(hypothesis)):
    print(hypothesis[i], ",", end=" ")
print(">")
