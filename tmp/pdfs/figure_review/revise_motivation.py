from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import math, hashlib, json

SRC=Path('D:/code/paper/motivation.vsdx')
OUT=Path('D:/code/paper/motivation_revised.vsdx')
N='http://schemas.microsoft.com/office/visio/2012/main'
ns={'v':N}
def tag(t): return '{'+N+'}'+t
def cell(p,n,v,f=None):
    el=p.find('v:Cell[@N="'+n+'"]',ns)
    if el is None: el=E.SubElement(p,tag('Cell'),N=n)
    el.attrib.clear(); el.set('N',n); el.set('V',str(v))
    if f is not None: el.set('F',f)
    return el
def sec(p,n): return E.SubElement(p,tag('Section'),N=n)
def row(p,i,t=None):
    a={'IX':str(i)}
    if t:a['T']=t
    return E.SubElement(p,tag('Row'),**a)
with ZipFile(SRC) as z: data={n:z.read(n) for n in z.namelist()}
root=E.fromstring(data['visio/pages/page1.xml'])
group=root.find('v:Shapes/v:Shape',ns)
shapes=group.find('v:Shapes',ns)
# Retain the original photograph, five caption quotations, and panel (a).
for s in list(shapes):
    if s.get('ID') not in ('1','2','5','6'): shapes.remove(s)
    else:
        for c in s.findall('v:Cell',ns):
            if 'Sheet.88!' in c.get('F',''):c.attrib.pop('F')
        c=s.find('v:Cell[@N="PinY"]',ns)
        cell(s,'PinY',float(c.get('V'))+.60)
W=5.118110492454436; H=3.355905617240595
for n,v in {'Width':W,'Height':H,'PinX':W/2+.08,'PinY':H/2+.08,'LocPinX':W/2,'LocPinY':H/2}.items():cell(group,n,v)
for c in root.findall('v:Connects',ns):root.remove(c)
next_id=100
def base(name,x,y,w,h,fill=None,line=None,lw=.7,trans=0,angle=0,dash=1):
    global next_id
    next_id+=1
    s=E.SubElement(shapes,tag('Shape'),ID=str(next_id),NameU=name,Type='Shape',LineStyle='0',FillStyle='0',TextStyle='0')
    values={'PinX':x,'PinY':y,'Width':w,'Height':h,'LocPinX':w/2,'LocPinY':h/2,'Angle':angle,'FlipX':0,'FlipY':0,'ResizeMode':0,
      'FillPattern':int(fill is not None),'FillForegnd':fill or '#FFFFFF','FillForegndTrans':trans,'FillBkgndTrans':trans,
      'LinePattern':dash if line else 0,'LineColor':line or '#000000','LineWeight':lw/72,'LineColorTrans':0,
      'ShdwPattern':0,'ShapeShdwShow':0,'FillGradientEnabled':0,'LineGradientEnabled':0,'QuickStyleVariation':0,'Rounding':0,
      'LeftMargin':0,'RightMargin':0,'TopMargin':0,'BottomMargin':0,'VerticalAlign':1}
    for n,v in values.items():cell(s,n,v)
    return s
def geom(s):
    g=sec(s,'Geometry');g.set('IX','0')
    for n,v in {'NoFill':0,'NoLine':0,'NoShow':0,'NoSnap':0}.items():cell(g,n,v)
    return g
def rect(name,x,y,w,h,line='#222222',fill=None,rounding=.12,lw=.75):
    s=base(name,x,y,w,h,fill,line,lw);cell(s,'Rounding',rounding)
    g=geom(s)
    for i,(a,b) in enumerate([(0,0),(1,0),(1,1),(0,1),(0,0)],1):
        r=row(g,i,'RelMoveTo' if i==1 else 'RelLineTo');cell(r,'X',a);cell(r,'Y',b)
    return s
def ellipse(name,x,y,w,h,fill=None,line=None,trans=0,angle=0,lw=.7,dash=1):
    s=base(name,x,y,w,h,fill,line,lw,trans,angle,dash)
    g=geom(s);r=row(g,1,'Ellipse')
    for n,v,f in [('X',w/2,'Width*0.5'),('Y',h/2,'Height*0.5'),('A',w,'Width'),('B',h/2,'Height*0.5'),('C',w/2,'Width*0.5'),('D',h,'Height')]:cell(r,n,v,f)
    return s
def line(name,x1,y1,x2,y2,color='#222222',lw=.7,start=0,end=0,dash=1):
    w=math.hypot(x2-x1,y2-y1);s=base(name,(x1+x2)/2,(y1+y2)/2,w,0,None,color,lw,angle=math.atan2(y2-y1,x2-x1),dash=dash)
    for n,v in {'BeginArrow':start,'EndArrow':end,'BeginArrowSize':0,'EndArrowSize':0}.items():cell(s,n,v)
    g=geom(s);cell(g,'NoFill',1)
    r=row(g,1,'MoveTo');cell(r,'X',0);cell(r,'Y',0)
    r=row(g,2,'LineTo');cell(r,'X',w,'Width');cell(r,'Y',0)
    return s
def text(name,txt,x,y,w,h=.16,size=7.8,color='#222222',bold=False,italic=False,align=1):
    s=base(name,x,y,w,h)
    c=sec(s,'Character');r=row(c,0)
    for n,v in {'Font':'Times New Roman','ComplexScriptFont':'Times New Roman','Color':color,'Size':size/72,'Style':int(bold)+2*int(italic),'LangID':'en-US'}.items():cell(r,n,v)
    p=sec(s,'Paragraph');r=row(p,0);cell(r,'HorzAlign',align);cell(r,'SpLine',-1)
    t=E.SubElement(s,tag('Text'));E.SubElement(t,tag('cp'),IX='0').tail=txt
    return s
BLUE='#00A9E0';PINK='#FF7492';RED='#F01942';ORANGE='#FF8A45';GREEN='#42B983'
PW=1.6732285; xs=[0,1.722441,3.444882]
for i,x in enumerate(xs):
    rect('Panel_'+str(i+2),x+PW/2,1.305,PW,1.55)
    line('Horizontal_axis_'+str(i),x+.16,.92,x+1.53,.92,lw=.8,end=4)
    line('Vertical_axis_'+str(i),x+.16,.92,x+.16,1.85,lw=.8,end=4)
    labels=['(b) Deterministic Representations','(c) Probabilistic Representations','(d) MCDisp-Align (Ours)']
    text('Panel_label_'+str(i),labels[i],x+PW/2,.65,PW-.04,size=7.6)
text('Distortion_label','Semantic distortion',xs[0]+PW/2,1.955,1.5,size=8.3,italic=True)
text('Collapse_label','Loss of semantic coverage',xs[1]+PW/2,1.955,1.5,size=8.0,italic=True)
text('Ours_label','Caption-set guidance',xs[2]+PW/2,1.955,1.5,size=8.3,italic=True)
# (b): retain the five-to-one point alignment mechanism.
center=(.81,1.34)
for i,(x,y) in enumerate([(.44,1.61),(.84,1.76),(1.29,1.59),(.40,1.11),(1.21,1.09)]):
    dx,dy=center[0]-x,center[1]-y;d=math.hypot(dx,dy)
    line('Point_alignment_'+str(i),x,y,center[0]-.035*dx/d,center[1]-.035*dy/d,end=5,lw=.7)
    ellipse('Caption_point_b_'+str(i),x,y,.054,.054,RED)
ellipse('Image_point_b',*center,.062,.062,BLUE)
text('Distortion_note','Distinct captions pulled to one point',xs[0]+PW/2,.805,1.55,size=6.8)

# (c) and (d) use the same caption-center geometry for direct comparison.
theta=math.radians(16)
pts=[(-.44,0),(-.23,.11),(0,-.14),(.23,.11),(.44,0)]
def rotate(x,y):return (x*math.cos(theta)-y*math.sin(theta),x*math.sin(theta)+y*math.cos(theta))
def pos(panel,x,y):
    a,b=rotate(x,y);return xs[panel]+.88+a,1.395+b
cx,cy=pos(1,0,0)
ellipse('Broader_image_range_c',cx,cy,1.27,.65,line=BLUE,lw=.8,angle=theta,dash=2)
for i,(a,b) in enumerate(pts):
    x,y=pos(1,a,b)
    ellipse('Individual_caption_distribution_c_'+str(i),x,y,.24,.155,PINK,trans=.45,angle=theta)
sx,sy=pos(1,-.32,.045)
ellipse('Contracted_image_distribution_c',sx,sy,.43,.30,BLUE,line=BLUE,trans=.50,lw=.65,angle=theta)
for i,(a,b) in enumerate(pts):
    x,y=pos(1,a,b);ellipse('Caption_center_c_'+str(i),x,y,.047,.047,RED)
ellipse('Image_center_c',sx,sy-.035,.053,.053,BLUE)
line('Contraction_arrow_c',cx+.25,cy+.24,sx+.10,sy+.17,color='#555555',lw=.65,end=5)
text('Collapse_note','Dashed: broader image range',xs[1]+PW/2,.805,1.55,size=6.8)

cx,cy=pos(2,0,0)
ellipse('Caption_set_distribution_d',cx,cy,1.27,.65,ORANGE,line=ORANGE,trans=.83,lw=.9,angle=theta)
ellipse('Image_distribution_d',cx,cy,1.16,.54,BLUE,line=BLUE,trans=.67,lw=.75,angle=theta)
# A separate unrelated semantic set remains outside the learned coverage.
ellipse('Unrelated_semantic_set_d',xs[2]+1.405,1.045,.23,.13,GREEN,trans=.50)
ellipse('Unrelated_center_d',xs[2]+1.405,1.045,.035,.035,GREEN)
a,b=pos(2,-.48,0);c,d=pos(2,.48,0)
line('Principal_direction_d',a,b,c,d,color='#344A5A',lw=.65,start=5,end=5)
for i,(a,b) in enumerate(pts):
    x,y=pos(2,a,b);ellipse('Caption_center_d_'+str(i),x,y,.049,.049,RED)
ellipse('Image_center_d',cx,cy,.056,.056,BLUE)
text('Direction_label','Principal direction',xs[2]+.88,1.77,1.30,size=7.0,color='#344A5A')
text('Ours_note','Preserve coverage and caption diversity',xs[2]+PW/2,.805,1.59,size=6.6)

# Two rows keep the individual-caption and caption-set legends distinct.
rect('Legend_frame',W/2,.275,W,.40,rounding=.08,lw=.65)
legend=[(.14,.37,'point',RED,'Caption center'),(1.73,.37,'point',BLUE,'Image center'),(3.40,.37,'ellipse',PINK,'Individual-caption distribution'),
        (.14,.18,'ellipse',BLUE,'Image distribution'),(1.73,.18,'ellipse',ORANGE,'Caption-set distribution'),(3.40,.18,'ellipse',GREEN,'Unrelated semantic set')]
for i,(x,y,kind,color,label) in enumerate(legend):
    ellipse('Legend_symbol_'+str(i),x,y,.052 if kind=='point' else .16,.052 if kind=='point' else .09,color,trans=0 if kind=='point' else .5)
    start=x+.11;right=[1.61,3.24,5.02][i%3]
    text('Legend_text_'+str(i),label,(start+right)/2,y,right-start,.14,size=7.1,align=0)

pages=E.fromstring(data['visio/pages/pages.xml']);pg=pages.find('v:Page',ns);ps=pg.find('v:PageSheet',ns)
for n,v in {'PageWidth':W+.16,'PageHeight':H+.16,'DrawingResizeType':0,'XRulerOrigin':0,'YRulerOrigin':0,'XGridOrigin':0,'YGridOrigin':0}.items():cell(ps,n,v)
pg.set('ViewScale','1.7');pg.set('ViewCenterX',str((W+.16)/2));pg.set('ViewCenterY',str((H+.16)/2))
data['visio/pages/page1.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True)
data['visio/pages/pages.xml']=E.tostring(pages,encoding='UTF-8',xml_declaration=True)
assert len({s.get('ID') for s in root.findall('.//v:Shape',ns)})==len(root.findall('.//v:Shape',ns))
for name,b in data.items():
    if name.endswith('.xml'):E.fromstring(b)
if OUT.exists():raise FileExistsError(OUT)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for name,b in data.items():z.writestr(name,b)
record={'source':str(SRC),'sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'output':str(OUT),'native_shapes':len(root.findall('.//v:Shape',ns))}
Path(__file__).with_name('revision_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
