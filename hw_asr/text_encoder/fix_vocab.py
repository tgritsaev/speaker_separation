fin = open("librispeech-vocab.txt", "r")
fout = open("ibrispeech-fixed-vocab.txt", "w+")

while (line := fin.readline()):
    line = line.lower().replace("'", "")
    print(line, end="", file=fout)
    
fin.close()
fout.close()