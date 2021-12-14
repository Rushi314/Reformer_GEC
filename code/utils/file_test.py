

file = open("/Users/rushi314/PycharmProjects/GrammerErrorCorrection/Reformer_GEC/code/data/bea19.txt")
file_contents = file.read()
contents_split = file_contents.splitlines()

write = open("pred.txt", "w")
for content in contents_split:
    write.write(content+'\n')
