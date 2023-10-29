fin = open("/home/mkairov/biology_research/dla_hw1/hw_asr/text_encoder/librispeech-vocab.txt", "r")
fout = open("/home/mkairov/biology_research/dla_hw1/hw_asr/text_encoder/librispeech-fixed-vocab.txt", "w+")

while (line := fin.readline()):
    line = line.lower().replace("'", "")
    print(line, end="", file=fout)
    
fin.close()
fout.close()