from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import math, hashlib, json, io
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

BASE=Path('D:/code/paper'); SRC=BASE/'framework.vsdx'
OUT=BASE/'framework_simplified.vsdx'; PDF=BASE/'framework_simplified.pdf'
TMP=Path(__file__).parent
N='http://schemas.microsoft.com/office/visio/2012/main'; ns={'v':N}
def tag(s):return '{'+N+'}'+s
def cell(p,n,v,f=None):
    c=p.find('v:Cell[@N="'+n+'"]',ns)
    if c is None:c=E.SubElement(p,tag('Cell'),N=n)
    c.attrib.clear();c.set('N',n);c.set('V',str(v))
    if f is not None:c.set('F',f)
    return c
def row(p,i,t=None):
    args={'IX':str(i)}
    if t:args['T']=t
    return E.SubElement(p,tag('Row'),**args)
def section(p,n):return E.SubElement(p,tag('Section'),N=n)
with ZipFile(SRC) as z:data={n:z.read(n) for n in z.namelist()}
source_root=E.fromstring(data['visio/pages/page1.xml'])
photo=deepcopy(source_root.find('.//v:Shape[@ID="9"]',ns))
root=E.Element(tag('PageContents'),nsmap={None:N,'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'})
shapes=E.SubElement(root,tag('Shapes'))
W,H=12,5.55
for key,file in [('TR','times.ttf'),('TB','timesbd.ttf'),('TI','timesi.ttf')]:
    pdfmetrics.registerFont(TTFont(key,'C:/Windows/Fonts/'+file))
pdf=canvas.Canvas(str(PDF),pagesize=(W*72,H*72))
pdf.setTitle('MCDisp-Align: simplified framework and data flow')
idx=1000;spec=[]
C={'ink':'#243246','muted':'#63758A','border':'#B7C2D0','blue':'#2389E8','bluebg':'#D6EAFF','pink':'#EA557D','pinkbg':'#FDDCE6','orange':'#ED8B41','orangebg':'#FCE5D0','purple':'#7C6AC1','gray':'#C8CED5'}
def color(s):return tuple(int(s.lstrip('#')[i:i+2],16)/255 for i in (0,2,4))
def style(fill=None,line=None,lw=1,alpha=1,dash=False):
    pdf.setFillAlpha(alpha);pdf.setStrokeAlpha(1);pdf.setLineWidth(lw)
    if fill:pdf.setFillColorRGB(*color(fill))
    if line:pdf.setStrokeColorRGB(*color(line))
    pdf.setDash([4,3] if dash else [])
def base(name,x,y,w,h,fill=None,line=None,lw=1,alpha=1,angle=0,dash=False):
    global idx;idx+=1
    s=E.SubElement(shapes,tag('Shape'),ID=str(idx),NameU=name,Name=name,IsCustomNameU='1',IsCustomName='1',Type='Shape',LineStyle='0',FillStyle='0',TextStyle='0')
    vals={'PinX':x,'PinY':y,'Width':w,'Height':h,'LocPinX':w/2,'LocPinY':h/2,'Angle':angle,'FlipX':0,'FlipY':0,'ResizeMode':0,
      'FillPattern':int(fill is not None),'FillForegnd':fill or '#FFFFFF','FillForegndTrans':1-alpha,'FillBkgndTrans':1-alpha,
      'LinePattern':(2 if dash else 1) if line else 0,'LineColor':line or C['ink'],'LineWeight':lw/72,'ShdwPattern':0,'ShapeShdwShow':0,
      'FillGradientEnabled':0,'LineGradientEnabled':0,'QuickStyleVariation':0,'Rounding':0,'LeftMargin':0,'RightMargin':0,'TopMargin':0,'BottomMargin':0,'VerticalAlign':1}
    for n,v in vals.items():cell(s,n,v)
    spec.append({'id':idx,'name':name,'x':x,'y':y,'w':w,'h':h})
    return s
def geom(s):
    g=section(s,'Geometry');g.set('IX','0')
    for n,v in {'NoFill':0,'NoLine':0,'NoShow':0,'NoSnap':0}.items():cell(g,n,v)
    return g
def rect(name,x,y,w,h,fill=None,line=None,r=.10,lw=1):
    s=base(name,x,y,w,h,fill,line,lw);cell(s,'Rounding',r)
    g=geom(s)
    for i,(a,b) in enumerate([(0,0),(1,0),(1,1),(0,1),(0,0)],1):
        q=row(g,i,'RelMoveTo' if i==1 else 'RelLineTo');cell(q,'X',a);cell(q,'Y',b)
    pdf.saveState();style(fill,line,lw);pdf.roundRect((x-w/2)*72,(y-h/2)*72,w*72,h*72,r*72,fill=bool(fill),stroke=bool(line));pdf.restoreState()
    return s
def ellipse(name,x,y,w,h,fill=None,line=None,alpha=1,angle=0,lw=1):
    s=base(name,x,y,w,h,fill,line,lw,alpha,angle);g=geom(s);q=row(g,1,'Ellipse')
    for n,v,f in [('X',w/2,'Width*0.5'),('Y',h/2,'Height*0.5'),('A',w,'Width'),('B',h/2,'Height*0.5'),('C',w/2,'Width*0.5'),('D',h,'Height')]:cell(q,n,v,f)
    pdf.saveState();style(fill,line,lw,alpha);pdf.translate(x*72,y*72);pdf.rotate(angle*180/math.pi);pdf.ellipse(-w*36,-h*36,w*36,h*36,fill=bool(fill),stroke=bool(line));pdf.restoreState()
    return s
def polygon(name,pts,fill,line=None,lw=1):
    lo=min(x for x,y in pts);hi=max(x for x,y in pts);bot=min(y for x,y in pts);top=max(y for x,y in pts)
    w=max(hi-lo,.001);h=max(top-bot,.001);s=base(name,(lo+hi)/2,(bot+top)/2,w,h,fill,line,lw);g=geom(s)
    for i,(x,y) in enumerate(pts+[pts[0]],1):
        q=row(g,i,'RelMoveTo' if i==1 else 'RelLineTo');cell(q,'X',(x-lo)/w);cell(q,'Y',(y-bot)/h)
    pdf.saveState();style(fill,line,lw);p=pdf.beginPath();p.moveTo(pts[0][0]*72,pts[0][1]*72)
    for x,y in pts[1:]:p.lineTo(x*72,y*72)
    p.close();pdf.drawPath(p,fill=bool(fill),stroke=bool(line));pdf.restoreState()
    return s
def arrow(name,pts,col=None,lw=1.8,end=True,start=False,dash=False):
    col=col or C['ink']
    for j,((x1,y1),(x2,y2)) in enumerate(zip(pts,pts[1:])):
        length=math.hypot(x2-x1,y2-y1)
        s=base(name+'_segment_'+str(j),(x1+x2)/2,(y1+y2)/2,length,0,None,col,lw,angle=math.atan2(y2-y1,x2-x1),dash=dash)
        g=geom(s);cell(g,'NoFill',1)
        q=row(g,1,'MoveTo');cell(q,'X',0);cell(q,'Y',0)
        q=row(g,2,'LineTo');cell(q,'X',length,'Width');cell(q,'Y',0)
        pdf.saveState();style(line=col,lw=lw,dash=dash);pdf.line(x1*72,y1*72,x2*72,y2*72);pdf.restoreState()
    def head(tip,other,name):
        x,y=tip;dx=x-other[0];dy=y-other[1];n=math.hypot(dx,dy);dx/=n;dy/=n
        l=.095;hw=.038
        polygon(name,[(x,y),(x-l*dx+hw*dy,y-l*dy-hw*dx),(x-l*dx-hw*dy,y-l*dy+hw*dx)],col)
    if end:head(pts[-1],pts[-2],name+'_arrowhead')
    if start:head(pts[0],pts[1],name+'_start_arrowhead')
def text(name,txt,x,y,w,h=.27,size=16,col=None,bold=False,italic=False,align=1):
    col=col or C['ink'];s=base(name,x,y,w,h)
    ch=section(s,'Character');q=row(ch,0)
    for n,v in {'Font':'Times New Roman','ComplexScriptFont':'Times New Roman','Color':col,'Size':size/72,'Style':int(bold)+2*int(italic),'LangID':'en-US'}.items():cell(q,n,v)
    para=section(s,'Paragraph');q=row(para,0);cell(q,'HorzAlign',align);cell(q,'SpLine',-1)
    for n,v,f in [('TxtPinX',w/2,'Width*0.5'),('TxtPinY',h/2,'Height*0.5'),('TxtWidth',w,'Width'),('TxtHeight',h,'Height'),('TxtLocPinX',w/2,'TxtWidth*0.5'),('TxtLocPinY',h/2,'TxtHeight*0.5')]:cell(s,n,v,f)
    tx=E.SubElement(s,tag('Text'));E.SubElement(tx,tag('cp'),IX='0').tail=txt
    font='TB' if bold else 'TI' if italic else 'TR'; lines=txt.split('\n');leading=size*1.08
    pdf.saveState();style(fill=col);pdf.setFont(font,size)
    for k,t in enumerate(lines):
        tw=pdfmetrics.stringWidth(t,font,size)
        if tw>w*72+.6:raise ValueError(f'Text too wide: {name}: {tw:.1f} > {w*72:.1f}')
        yy=y*72+(len(lines)-1)*leading/2-k*leading-size*.32
        if align==1:pdf.drawCentredString(x*72,yy,t)
        elif align==0:pdf.drawString((x-w/2)*72,yy,t)
    pdf.restoreState();return s
def formula(name,runs,x,y,w,size=18):
    # Runs use baseline/superscript/subscript and remain editable native Visio text.
    s=text(name,'',x,y,w,.38,size=size);ch=s.find('v:Section[@N="Character"]',ns)
    for pos in (1,2):
        q=deepcopy(ch[0]);q.set('IX',str(pos));cell(q,'Pos',pos);ch.append(q)
    tx=s.find('v:Text',ns);tx.clear()
    widths=[pdfmetrics.stringWidth(t,'TR',size if p==0 else size*.65) for t,p in runs]
    xx=x*72-sum(widths)/2
    if sum(widths)>w*72:raise ValueError(name+' formula too wide')
    pdf.saveState();style(fill=C['ink'])
    for (t,p),tw in zip(runs,widths):
        E.SubElement(tx,tag('cp'),IX=str(p)).tail=t
        sz=size if p==0 else size*.65;pdf.setFont('TR',sz)
        pdf.drawString(xx,y*72-size*.32+(size*.42 if p==1 else -size*.22 if p==2 else 0),t);xx+=tw
    pdf.restoreState()
def gaussian(name,x,y,w,h,col,angle=.35,center=True,star=False):
    for j,(scale,alpha) in enumerate([(1,.19),(.73,.19),(.43,.21)]):ellipse(name+'_contour_'+str(j),x,y,w*scale,h*scale,col,alpha=alpha,angle=angle)
    if center:
        if star:
            pts=[]
            for j in range(10):
                th=math.pi/2+j*math.pi/5;r=.095 if j%2==0 else .043;pts.append((x+r*math.cos(th),y+r*math.sin(th)))
            polygon(name+'_center',pts,col)
        else:ellipse(name+'_center',x,y,.062,.062,col)

# Three compact modules, with one common color system and explicit routing.
rect('Module_1',2.055,2.725,3.95,4.45,'#FFFFFF',C['border'],r=.13)
rect('Module_2',6.085,2.725,3.21,4.45,'#FFFFFF',C['border'],r=.13)
rect('Module_3',10.06,2.725,3.72,4.45,'#FFFFFF',C['border'],r=.13)
text('Title_1','1  Probabilistic representation',2.055,4.71,3.77,size=18,bold=True)
text('Title_2','2  Caption-set construction',6.085,4.71,3.05,size=17.5,bold=True)
text('Title_3','3  Matching & alignment',10.06,4.71,3.55,size=18,bold=True)

# Original photograph, retained as an editable Visio image object.
photo.set('ID','9');photo.set('NameU','Original_image');photo.set('Name','Original image')
for n,v in {'PinX':.72,'PinY':3.68,'Width':1.03,'Height':.913,'LocPinX':.515,'LocPinY':.4565,'ImgWidth':1.03,'ImgHeight':.913,'ImgOffsetX':0,'ImgOffsetY':0}.items():cell(photo,n,v)
for c in photo.findall('v:Cell',ns):
    if 'Sheet.' in c.get('F',''):c.attrib.pop('F')
layer=photo.find('v:Cell[@N="LayerMember"]',ns)
if layer is not None:photo.remove(layer)
shapes.append(photo)
pdf.drawImage(ImageReader(io.BytesIO(data['visio/media/image1.png'])),(.72-.515)*72,(3.68-.4565)*72,1.03*72,.913*72)
text('Input_image','Image',.72,3.07,1.03,size=16)
rect('Vision_encoder_head',2.12,3.68,1.38,.85,C['bluebg'],C['blue'],r=.08)
text('Vision_label','Vision encoder\nGaussian head',2.12,3.68,1.35,.62,size=15.8)
arrow('Image_to_encoder',[(1.26,3.68),(1.40,3.68)],lw=1.4)
arrow('Encoder_to_image_distribution',[(2.84,3.68),(3.03,3.68)],lw=1.4)
gaussian('Image_distribution',3.50,3.68,.85,.55,C['blue'])
formula('Image_covariance',[('Σ',0),('v',1),(' = Diag(v',0),('v',1),(') + UU',0),('T',1)],2.24,2.83,3.34,size=18)

for j,label in enumerate(['Carriage','Horses','Tourists']):
    yy=2.12-j*.31;rect('Caption_card_'+str(j),.72,yy,1.08,.27,C['pinkbg'],None,r=.045)
    text('Caption_excerpt_'+str(j),label,.72,yy,.98,.23,size=14.8)
text('Caption_input','Captions',.72,1.10,1.15,size=16)
rect('Text_encoder_head',2.12,1.81,1.38,.85,C['pinkbg'],C['pink'],r=.08)
text('Text_label','Text encoder\nGaussian head',2.12,1.81,1.35,.62,size=15.8)
arrow('Captions_to_encoder',[(1.29,1.81),(1.40,1.81)],lw=1.4)
arrow('Encoder_to_caption_distributions',[(2.84,1.81),(3.02,1.81)],lw=1.4)
for j,(dx,dy) in enumerate([(-.21,.16),(.16,.19),(-.24,-.18),(.18,-.15),(0,0)]):gaussian('Caption_distribution_'+str(j),3.49+dx,1.81+dy,.31,.19,C['pink'],center=True)
text('Caption_covariance_label','Diagonal',3.49,1.10,1.0,size=16)
text('Trainable_note','Trainable encoders + heads',2.05,.73,3.7,size=15.5,col=C['muted'])

# Moment matching creates the set Gaussian; the lower strip shows its statistics.
rect('Moment_matching',5.34,3.71,1.24,.76,'#F1EAFE',C['purple'],r=.08)
text('Moment_matching_label','Moment\nmatching',5.34,3.71,1.13,.60,size=17,bold=True)
arrow('Moment_to_set',[(5.99,3.71),(6.24,3.71)],C['orange'])
gaussian('Caption_set_distribution',6.84,3.71,1.00,.67,C['orange'],star=True)
text('Set_distribution_label','Caption set',6.84,3.15,1.3,size=16)
text('Statistics_heading','Caption statistics',6.085,2.79,2.9,size=16.5,bold=True)
# Mean icon.
for j,(dx,dy) in enumerate([(-.21,.08),(.19,.10),(-.06,-.13)]):ellipse('Mean_point_'+str(j),5.11+dx,2.19+dy,.055,.055,C['pink'])
ellipse('Set_mean_icon',5.11,2.19,.086,.086,C['orange'])
# Range icon.
gaussian('Set_range_icon',6.09,2.19,.67,.42,C['orange'],center=False)
# Center variation direction icon; no full-covariance eigenvector claim.
arrow('Center_direction_icon',[(6.77,1.99),(7.39,2.38)],C['orange'],lw=1.3,start=True)
for j,(dx,dy) in enumerate([(-.19,-.08),(.01,-.035),(.15,.12)]):ellipse('Direction_caption_point_'+str(j),7.09+dx,2.19+dy,.050,.050,C['pink'])
text('Center_label','Center',5.11,1.64,.83,size=16)
text('Range_label','Range',6.09,1.64,.83,size=16)
text('Directions_label','Directions',7.09,1.64,1.02,size=16)
formula('Caption_set_covariance',[('Σ',0),('C',1),(' = mean',0),('k',2),('(Σ',0),('t',1),('k',2),(') + Cov',0),('k',2),('(μ',0),('t',1),('k',2),(')',0)],6.085,1.03,3.03,size=17)

# Matching is separate from geometric alignment; only the latter detaches targets.
rect('Matching_block',10.06,3.74,3.30,1.24,'#F8FAFD',C['border'],r=.08,lw=.8)
text('Matching_heading','Bidirectional matching',10.06,4.12,3.03,size=17,bold=True)
gaussian('Matching_image',9.07,3.60,.68,.42,C['blue'])
gaussian('Matching_positive_set',10.18,3.60,.70,.43,C['orange'],star=True)
arrow('Positive_match',[(9.50,3.60),(9.73,3.60)],C['purple'],lw=1.5,start=True)
gaussian('Negative_set_1',11.01,3.72,.43,.26,C['gray'],star=True)
gaussian('Negative_set_2',11.37,3.52,.38,.23,C['gray'],star=True)
text('Negative_label','Negatives',11.17,3.23,.96,size=14.7,col=C['muted'])

rect('Geometric_alignment_block',10.06,1.99,3.30,1.85,'#FCFBF8',C['border'],r=.08,lw=.8)
text('Geometric_alignment_heading','Geometric alignment',10.06,2.63,3.02,size=17,bold=True)
gaussian('Align_center_image',8.98,2.04,.54,.37,C['blue'],center=False)
gaussian('Align_center_set',9.15,2.10,.54,.37,C['orange'],center=False)
text('Center_base_variance_label','Center &\nbase variance',9.08,1.48,1.12,.49,size=14.6)
gaussian('Coverage_image',10.17,2.08,.75,.46,C['blue'],center=False)
for j,(dx,dy) in enumerate([(-.22,-.03),(-.08,.11),(.12,.02),(.22,.08),(.03,-.12)]):ellipse('Covered_center_'+str(j),10.17+dx,2.08+dy,.047,.047,C['pink'])
text('Coverage_label','Coverage',10.17,1.59,.92,size=15.7)
formula('Coverage_boundary',[('d',0),('2',1),(' ≤ D',0)],10.17,1.28,.86,size=15.5)
arrow('Direction_target',[(10.86,1.88),(11.45,2.25)],C['orange'],lw=1.5,start=True)
arrow('Direction_image',[(10.88,1.83),(11.47,2.20)],C['blue'],lw=1.5,start=True)
text('Direction_alignment_label','Directions',11.19,1.55,1.00,size=15.7)
text('Variance_regularization_label','Variance regularization',10.22,.77,2.6,size=15.4,col=C['muted'])
ellipse('Image_variance_regularizer',8.62,.77,.13,.08,C['blue'])
ellipse('Caption_variance_regularizer',8.82,.77,.13,.08,C['pink'])

# Inter-module data flow. Routes have separate ports and no ambiguous crossings.
arrow('Image_distribution_to_matching',[(3.95,3.68),(4.12,3.68),(4.12,5.20),(8.04,5.20),(8.04,4.45),(9.07,4.45),(9.07,4.38)],C['blue'],lw=2.1)
text('Image_route_label','Image distribution',6.05,5.37,2.95,size=16.5,col=C['blue'])
arrow('Caption_distributions_to_construction',[(3.93,1.81),(4.25,1.81),(4.25,3.71),(4.69,3.71)],C['pink'],lw=2.1)
arrow('Set_distribution_to_matching',[(7.38,3.71),(8.39,3.71)],C['orange'],lw=2.1)
# The statistics port represents both individual centers and set-derived targets.
arrow('Caption_statistics_to_alignment',[(7.70,2.23),(8.39,2.23)],C['orange'],lw=2.1)
rect('Stop_gradient_marker',8.04,2.23,.11,.18,'#FFFFFF',C['orange'],r=0,lw=1.5)
text('Stop_gradient_label','sg',8.04,2.50,.31,.21,size=14,col=C['orange'],bold=True)
# The image parameters also feed alignment, bypassing the matching operation.
arrow('Image_distribution_to_alignment',[(9.07,4.45),(11.82,4.45),(11.82,2.99),(9.07,2.99),(9.07,2.94)],C['blue'],lw=1.65)
ellipse('Image_branch_junction',9.07,4.45,.054,.054,C['blue'])

# Minimal legend: color identifies modality; dots and stars denote centers.
for x,col,label in [(2.45,C['blue'],'Image'),(4.20,C['pink'],'Caption'),(6.32,C['orange'],'Caption set')]:
    ellipse('Legend_'+label.replace(' ','_'),x,.24,.13,.08,col)
    text('Legend_label_'+label.replace(' ','_'),label,x+.67,.24,1.10,.23,size=15.5,align=0)
text('Stop_gradient_key','sg: stop-gradient',9.68,.24,2.10,.23,size=15.5,col=C['muted'])

pages=E.fromstring(data['visio/pages/pages.xml']);p=pages.find('v:Page',ns);ps=p.find('v:PageSheet',ns)
for n,v in {'PageWidth':W,'PageHeight':H,'DrawingResizeType':0,'XRulerOrigin':0,'YRulerOrigin':0,'XGridOrigin':0,'YGridOrigin':0}.items():cell(ps,n,v)
p.set('ViewCenterX',str(W/2));p.set('ViewCenterY',str(H/2));p.set('ViewScale','.85')
data['visio/pages/page1.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True)
data['visio/pages/pages.xml']=E.tostring(pages,encoding='UTF-8',xml_declaration=True)
assert OUT.name=='framework_simplified.vsdx' and OUT!=SRC
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for n,b in data.items():z.writestr(n,b)
pdf.showPage();pdf.save()
ids=[s.get('ID') for s in root.findall('.//v:Shape',ns)];assert len(ids)==len(set(ids))
record={'outputs':[str(OUT),str(PDF)],'source_hashes':{n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ('framework.vsdx','framework_backup_20260922.vsdx','iclr2027_conference.tex')},'native_shapes':len(ids),'formula_count':3,'shapes':spec}
(TMP/'framework_simplified_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='shapes'},indent=2))




