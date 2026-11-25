"""Build an A3 project roadmap from verified project results. No model inference."""
from pathlib import Path
import hashlib
import json
import re
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'comedic-timing-roadmap.pdf'
W, H = 1200, 1697
INK = '#16333A'
MUTED = '#536B70'
PAPER = '#F5F3EC'
GREEN = '#157C69'
AMBER = '#AF5D16'
BLUE = '#355E85'
RED = '#B85A48'
LINE = '#D6DEDA'
FONT = Path('C:/Windows/Fonts')
for name, path in [('Regular', 'segoeui.ttf'), ('Bold', 'segoeuib.ttf'), ('Light', 'segoeuil.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / path)))

summary = json.loads((ROOT/'runs/second-aligner-investigation-v1/summary.json').read_text())
verification = json.loads((ROOT/'runs/second-aligner-investigation-v1/verification.json').read_text())
assert summary['original_flags'] == 34 and summary['cleared_flags'] == 0
assert summary['onset_targets'] == 29 and summary['artifact_targets'] == 5
assert verification['fixture_constructions_verified'] == 384

c = canvas.Canvas(str(PDF), pagesize=A3, pageCompression=1)
c.setTitle('Comedic timing in voice models - research roadmap')
c.setAuthor('Comedic timing project')
c.setSubject('Current evidence, unresolved measurement questions and proposed local next steps; 15 November 2026')
c.scale(A3[0]/W, A3[1]/H)
layouts = []

def col(value):
    return colors.HexColor(value)

def box(x, y, w, h, fill, stroke=None, radius=18):
    c.setFillColor(col(fill))
    c.setStrokeColor(col(stroke or fill))
    c.setLineWidth(1.2)
    c.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=bool(stroke))

def text(x, y, s, size=22, font='Regular', fill=INK):
    c.setFillColor(col(fill)); c.setFont(font, size)
    c.drawString(x, H-y-size*.80, s)
    width=pdfmetrics.stringWidth(s,font,size)
    assert x+width <= W-40, (s,x,width)
    layouts.append({'text':s,'x':x,'y':y,'width':width,'height':size})

def para(x, y, s, width, size=21, leading=29, fill=MUTED, font='Regular'):
    lines=[]
    for paragraph in s.split('\n'):
        line=''
        for word in paragraph.split():
            trial=(line+' '+word).strip()
            if pdfmetrics.stringWidth(trial,font,size)>width and line:
                lines.append(line);line=word
            else:line=trial
        lines.append(line)
    for i,line in enumerate(lines):text(x,y+i*leading,line,size,font,fill)
    return y+len(lines)*leading

def rule(x1,y,x2,fill=LINE,width=2):
    c.setStrokeColor(col(fill));c.setLineWidth(width);c.line(x1,H-y,x2,H-y)

def pill(x,y,s,fg,bg,size=16):
    width=pdfmetrics.stringWidth(s,'Bold',size)+28
    box(x,y,width,31,bg,radius=15)
    text(x+14,y+8,s,size,'Bold',fg)

# Header: restrained editorial typography, with one waveform motif.
box(0,0,W,H,PAPER,radius=0)
text(66,48,'RESEARCH ROADMAP',19,'Bold',GREEN)
text(940,48,'15 NOV 2026',17,'Bold',MUTED)
text(66,100,'Comedic timing',65,'Bold')
text(66,174,'in voice models',65,'Bold')
text(68,264,'Can we measure the pause reliably before studying the joke?',25,'Regular',MUTED)

# Abstract speech / pause / speech motif. Decorative, not a data plot.
for i, amplitude in enumerate([18,35,59,42,25,48,67,30,18,7,0,0,0,0,0,8,24,56,73,45,29,59,34,14]):
    x=836+i*11
    c.setStrokeColor(col(GREEN if i<10 else AMBER if i<15 else BLUE));c.setLineWidth(4)
    c.line(x,H-(173-amplitude/2),x,H-(173+amplitude/2))
text(927,223,'the pause',16,'Regular',AMBER)

# Overall path. The second stage remains open; later stages are conditional.
rule(95,363,1105,LINE,3)
stages=[(66,'01','Diagnostic foundation','COMPLETE',GREEN,'#E5F0E9'),
        (342,'02','Validate measurement','YOU ARE HERE',AMBER,'#F7E4CD'),
        (618,'03','Broader model pilot','CONDITIONAL',BLUE,'#E5EDF3'),
        (894,'04','Main study','LATER',MUTED,'#E9EDE9')]
for x,num,label,status,fg,bg in stages:
    box(x,328,240,139,'#FFFFFF' if num=='02' else PAPER,AMBER if num=='02' else None,14)
    box(x+16,344,39,39,fg,radius=19)
    text(x+25,354,num,17,'Bold','#FFFFFF')
    assert pdfmetrics.stringWidth(label,'Bold',19) <= 208
    text(x+16,399,label,19,'Bold')
    text(x+16,433,status,14,'Bold',fg)

# Central status: all original flags preserved, with distinct evidence categories.
box(66,505,1068,310,INK,radius=22)
pill(90,529,'CURRENT POSITION / GATE 1 NOT ACCEPTED','#F1BC7A','#354447',15)
text(96,585,'34',91,'Bold','#FFFFFF')
text(216,619,'/ 40',44,'Light','#D6E1E0')
text(96,693,'clips remain flagged',26,'Bold','#FFFFFF')
para(96,737,'6 earlier duration-QC passes\nare not a validated study subset.',326,18,25,'#C5D5D4')
c.setStrokeColor(col('#506468'));c.setLineWidth(1);c.line(449,H-581,449,H-784)
text(480,583,'29',42,'Bold','#EDBA79')
text(565,585,'Onsets corroborated',26,'Bold','#FFFFFF')
para(565,626,'Charsiu / newer MFA: within 30 ms of original MFA.\nMuting reveals anticipation; accuracy is unproven.',530,19,27,'#D8E2E1')
rule(480,695,1104,'#506468',1)
text(484,720,'5',42,'Bold','#EDBA79')
text(565,721,'Artifacts localized',26,'Bold','#FFFFFF')
para(565,761,'Vocoder style affects baseline wander.\nChanges also alter speech; no validated repair.',530,19,27,'#D8E2E1')

# Scope counters. Distinguish scientific validation from software checks.
for x,value,title,detail in [(66,'40','diagnostic clips','10 pairs x 2 voices; one TTS family'),
                             (430,'384','edited-audio cases','Broader recovery tests; failures remain'),
                             (794,'23','software tests passed','Code checks do not prove accuracy')]:
    text(x,850,value,43,'Bold',GREEN)
    text(x,906,title,20,'Bold')
    text(x,937,detail,15,'Regular',MUTED)

text(66,995,'NEXT / LOCAL EXPERIMENTS',25,'Bold')
text(805,1001,'Proposed next extensions',18,'Regular',BLUE)

def next_card(x,y,n,title,body,deliverable):
    box(x,y,522,180,'#FFFFFF',LINE,16)
    box(x+21,y+21,33,33,'#E6EDF2',radius=10)
    text(x+31,y+29,n,17,'Bold',BLUE)
    text(x+68,y+25,title,23,'Bold')
    end=para(x+23,y+71,body,476,19,26)
    assert end <= y+152, (title,end)
    text(x+23,y+153,deliverable,16,'Bold',BLUE)

next_card(66,1043,'A',"Define what 'pause' means",
          'Separate phonetic silence from waveform quiet. Freeze thresholds and missing-value rules.',
          'OUTPUT  Versioned measurement specification')
next_card(612,1043,'B','Test context dependence',
          'Change future context, preserve boundary samples. Challenge new phones, scripts and input levels.',
          'OUTPUT  Map of onset failure conditions')
next_card(66,1247,'C','Check correction side effects',
          'Carry fixed corrections to unseen clips and phonetic conditions. Track speech loss and timing shifts.',
          'OUTPUT  Explicit accept/reject criteria for repairs')
next_card(612,1247,'D','Check transfer across models',
          'Add a distinct local TTS family. Repeat whole-script generation and the same measurement checks.',
          'OUTPUT  Replication beyond Kokoro')

# Conditional destination, without promising that local tests certify natural truth.
box(66,1461,1068,132,'#E7ECE7',radius=16)
text(89,1481,'AFTER MEASUREMENT ACCEPTANCE',17,'Bold',GREEN)
text(89,1514,'15-pair model pilot  >  larger Mandarin / English corpus  >  paired timing + surprisal',21,'Bold')
para(89,1551,'Target: 60 retained pairs per language. Independent natural-boundary accuracy remains an open requirement.',1021,18,25)

text(66,1622,'LOCAL INFERENCE ONLY   |   NO BILLED APIS   |   HUMAN ANNOTATION DEFERRED',17,'Bold',GREEN)
text(66,1654,'Evidence: PLAN.md  |  reports/second-aligner-investigation.md  |  saved integrity audit',15,'Regular',MUTED)
c.showPage();c.save()

# Text-level verification complements the required rendered-page review.
reader=PdfReader(PDF);assert len(reader.pages)==1
extracted=reader.pages[0].extract_text()
for required in ['34','384','23','accuracy is unproven','no validated repair','Proposed next extensions','GATE 1 NOT ACCEPTED','HUMAN ANNOTATION DEFERRED']:
    assert required in extracted, required
inputs=['PLAN.md','reports/second-aligner-investigation.md','runs/second-aligner-investigation-v1/summary.json','runs/second-aligner-investigation-v1/verification.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'comedic-timing-roadmap.provenance.json').write_text(json.dumps({
    'as_of_local_date':'2026-11-15','timezone':'America/Indianapolis','format':'A3 portrait, vector PDF',
    'inputs_sha256':{p:sha(ROOT/p) for p in inputs},'builder_sha256':sha(Path(__file__)),
    'pdf_sha256':sha(PDF),'pages':len(reader.pages),'text_checked':True,
    'poster_role':'Project status and proposed roadmap; no new scientific experiments or gate acceptance.'
},indent=2),encoding='utf-8')
print(PDF)
print('Verified one page and required status/caveat text.')
