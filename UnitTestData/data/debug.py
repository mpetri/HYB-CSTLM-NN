import collections

tr = []
for line in open("training.data").readlines():
    toks = map(int, line.strip().split())
    toks = [3] + toks + [4]
    tr.append(toks)

bigrams = collections.Counter()
for s in tr:
    for i in range(len(s)-1):
        bigrams[s[i], s[i+1]] += 1
        
print 'Bigram statistics'
print 'n1 ', len([key for key, value in bigrams.items() if value == 1])
print 'n2 ', len([key for key, value in bigrams.items() if value == 2])
print 'n3+', len([key for key, value in bigrams.items() if value >= 3])

unigrams = collections.Counter()
for s in tr:
    for i in range(1,len(s)):
        unigrams[s[i]] += 1

print 'Unigram statistics'
print 'n1 ', len([key for key, value in unigrams.items() if value >= 1])
print 'n2 ', len([key for key, value in unigrams.items() if value >= 2])
print 'n3+', len([key for key, value in unigrams.items() if value >= 3])

things = collections.Counter()
for (w1, w2), value in bigrams.items():
    things[w2] += 1
print 'Thing statistics'
print 'n1 ', len([key for key, value in things.items() if value >= 1])
print 'n2 ', len([key for key, value in things.items() if value >= 2])
print 'n3+', len([key for key, value in things.items() if value >= 3])
