idx2word = [0, 1, 2]
word2cnt = {0:13011, 1:2111110, 2:100}
table_size = 10

denom = 0
for word_idx in idx2word:
    denom += word2cnt[word_idx] ** 0.75
for word_idx in idx2word:
    word2cnt[word_idx] = (word2cnt[word_idx] ** 0.75) / denom

table = [0] * table_size
i = 1
d1 = word2cnt[i]
for a in range(table_size):
    table[a] = i
    if a / table_size > d1:
        i += 1
        if i >= len(idx2word):
            i = len(idx2word) - 1
        d1 += (word2cnt[i] ** 0.75) / denom

for a in table:
    print(a)
