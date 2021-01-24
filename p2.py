import csv

with open("p2.csv", "r") as f:
    reader = csv.reader(f)
    rows = list(reader)

S = rows[0][:-1]
nfeatures = len(S)
G_null = nfeatures * ["?"]
G = []

print("S0", S)
print("G0", G_null)
print("---")

for r, row in enumerate(rows):
    if row[nfeatures] == "Yes":
        for i in range(nfeatures):
            if S[i] != row[i]:
                S[i] = "?"

        for i in range(nfeatures):
            to_remove = []
            for j, g in enumerate(G):
                if g[i] != "?" and S[i] != g[i]:
                    to_remove.append(j)
            for j in to_remove:
                del G[j]

    elif row[nfeatures] == "No":
        for i in range(nfeatures):
            if S[i] != "?" and S[i] != row[i]:
                g = ["?"] * nfeatures
                g[i] = S[i]
                G.append(g)

    print(f"D{r + 1}", row)
    print("---")
    print(f"S{r + 1}", S)
    print(f"G{r + 1}", G if G else G_null)
    print("---")
