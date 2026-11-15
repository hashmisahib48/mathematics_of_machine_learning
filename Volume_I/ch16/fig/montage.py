import sys; from PIL import Image
out=sys.argv[1]; fs=sys.argv[2:]
ims=[Image.open(f).convert('RGB') for f in fs]
# scale each to 0.4 of native and stack in 2 columns
sc=[i.resize((int(i.width*0.42),int(i.height*0.42))) for i in ims]
cols=2; rows=(len(sc)+1)//2
cw=max(i.width for i in sc); rh=[max(i.height for i in sc[r*2:r*2+2]) for r in range(rows)]
m=Image.new('RGB',(cw*cols+10,sum(rh)+10*rows),'white'); y=0
for r in range(rows):
    for c in range(cols):
        k=r*2+c
        if k<len(sc): m.paste(sc[k],(c*(cw+10),y))
    y+=rh[r]+10
m.save(out)
