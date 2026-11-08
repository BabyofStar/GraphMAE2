"""Render the two report figures with ReportLab's chart toolkit (SVG outputs).

PNG conversion is handled by render_preview.cjs using Sharp.
"""
from pathlib import Path
import json
import re
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, white

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT/'tables/unified_results.json').read_text('utf-8'))
FIG = ROOT/'figures'
FIG.mkdir(exist_ok=True)
INK=HexColor('#172b40');MUTED=HexColor('#526779');GRID=HexColor('#dce4ed')
BLUE=HexColor('#2563a6');TEAL=HexColor('#087f83');ORANGE=HexColor('#b65e21')
COLORS=[BLUE,ORANGE,TEAL]
DS=['cora','citeseer','pubmed'];NAMES=['Cora','Citeseer','PubMed']


def label(d,x,y,text,size=12,color=INK,anchor='start',bold=False):
    d.add(String(x,y,text,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=color,textAnchor=anchor))


def axes(plot):
    for axis in [plot.xValueAxis,plot.yValueAxis]:
        axis.strokeColor=GRID
        axis.labels.fontName='Helvetica'
        axis.labels.fontSize=11
        axis.labels.fillColor=MUTED
    plot.xValueAxis.tickDown=4
    plot.yValueAxis.tickLeft=4
    plot.yValueAxis.visibleGrid=True
    plot.yValueAxis.gridStrokeColor=GRID
    plot.yValueAxis.gridStrokeWidth=.5


def save(drawing, stem):
    svg=renderSVG.drawToString(drawing)
    if isinstance(svg,bytes):svg=svg.decode('utf-8')
    svg=re.sub(r'<!DOCTYPE svg.*?>','',svg,flags=re.S)
    svg=svg.replace('font-family: Helvetica-Bold;', 'font-family: Arial, sans-serif; font-weight: 700;')
    svg=svg.replace('font-family: Helvetica;', 'font-family: Arial, sans-serif;')
    (FIG/(stem+'.svg')).write_text(svg,encoding='utf-8')


def formal():
    d=Drawing(1200,620)
    d.add(Rect(0,0,1200,620,fillColor=white,strokeColor=None))
    label(d,48,575,'Paired test-accuracy changes: B2 - B0',26,bold=True)
    label(d,48,547,'Dots: training seeds 10-19. Square and whisker: mean and unadjusted 95% paired t interval.',14,MUTED)
    ymin,ymax=-1.1,1.7
    for i,ds in enumerate(DS):
        p=DATA['formal_primary'][i];rows=[r for r in DATA['formal_per_seed'] if r['dataset']==ds]
        left=62+i*393;bottom=169;w=326;h=302
        lp=LinePlot();lp.x=left;lp.y=bottom;lp.width=w;lp.height=h
        lp.data=[[(r['seed'],r['delta_pp']) for r in rows]]
        lp.joinedLines=0
        lp.lines[0].symbol=makeMarker('FilledCircle',size=6.5,fillColor=COLORS[i],strokeColor=COLORS[i])
        lp.xValueAxis.valueMin=9.5;lp.xValueAxis.valueMax=22
        lp.xValueAxis.valueSteps=list(range(10,20))+[21]
        lp.xValueAxis.labelTextFormat=lambda value:'Mean' if value==21 else str(int(value))
        lp.yValueAxis.valueMin=ymin;lp.yValueAxis.valueMax=ymax
        lp.yValueAxis.valueSteps=[-1,-.5,0,.5,1,1.5]
        lp.yValueAxis.labelTextFormat=lambda value:f'{value:g}'
        axes(lp)
        d.add(lp)
        yp=lambda value:bottom+(value-ymin)/(ymax-ymin)*h
        xp=lambda value:left+(value-9.5)/(22-9.5)*w
        d.add(Line(left,yp(0),left+w,yp(0),strokeColor=MUTED,strokeWidth=1.1))
        x=xp(21);a=yp(p['ci95_unadjusted_low_pp']);b=yp(p['ci95_unadjusted_high_pp'])
        d.add(Line(x,a,x,b,strokeColor=INK,strokeWidth=2))
        for end in [a,b]:d.add(Line(x-6,end,x+6,end,strokeColor=INK,strokeWidth=2))
        d.add(Rect(x-4,yp(p['mean_delta_pp'])-4,8,8,fillColor=INK,strokeColor=INK))
        label(d,left,510,NAMES[i],19,bold=True)
        label(d,left,486,f"Mean {p['mean_delta_pp']:+.3f} pp | Holm p = {p['p_holm_three_datasets']:.4f}",12,MUTED)
        label(d,left+w/2,125,'Paired training seed',12,MUTED,'middle')
        label(d,left,99,f"Win / tie / loss: {p['wins']} / {p['ties']} / {p['losses']}",12,MUTED)
    label(d,48,65,'Vertical scale: test-accuracy difference in percentage points (pp); identical across panels.',13,MUTED)
    label(d,48,40,'Intervals use sample SD of paired differences (ddof=1). Only Cora passes Holm correction across 3 datasets.',13,MUTED)
    save(d,'formal_B2_minus_B0')


def mechanism():
    d=Drawing(1250,620)
    d.add(Rect(0,0,1250,620,fillColor=white,strokeColor=None))
    label(d,48,575,'S / R mechanism contrast on development validation',25,bold=True)
    label(d,48,547,'Exploratory evidence, n=3 training seeds; one fixed random grouping per graph.',14,MUTED)
    label(d,70,505,'A. Variation in true-density-layer masking rates',16,bold=True)
    label(d,70,480,'Temporal SD (pp), averaged over 4 layers and 3 seeds',12,MUTED)
    bc=VerticalBarChart();bc.x=77;bc.y=180;bc.width=485;bc.height=265
    bc.data=[[next(r['value_pp'] for r in DATA['mechanism'] if r['dataset']==ds and r['method']==m) for ds in DS] for m in ['B0','S','R']]
    bc.categoryAxis.categoryNames=NAMES
    bc.valueAxis.valueMin=0;bc.valueAxis.valueMax=2;bc.valueAxis.valueStep=.5
    bc.valueAxis.visibleGrid=True;bc.valueAxis.gridStrokeColor=GRID;bc.valueAxis.gridStrokeWidth=.5
    bc.categoryAxis.strokeColor=GRID;bc.valueAxis.strokeColor=GRID
    for axis in [bc.categoryAxis,bc.valueAxis]:
        axis.labels.fontName='Helvetica';axis.labels.fontSize=12;axis.labels.fillColor=MUTED
    bc.groupSpacing=25;bc.barSpacing=5
    for i,color in enumerate([BLUE,TEAL,ORANGE]):bc.bars[i].fillColor=color;bc.bars[i].strokeColor=None
    d.add(bc)
    for i,(name,color) in enumerate(zip(['B0','S','R'],[BLUE,TEAL,ORANGE])):
        x=155+i*110;d.add(Rect(x,125,12,12,fillColor=color,strokeColor=None));label(d,x+21,125,name,13)
    label(d,70,96,'S fixes true-density quotas, so its variation is zero.',12,MUTED)
    label(d,660,505,'B. S - R paired validation changes',16,bold=True)
    label(d,660,480,'Validation accuracy difference (pp); points are individual seeds',12,MUTED)
    plot=LinePlot();plot.x=686;plot.y=180;plot.width=490;plot.height=265
    diffs={ds:next(r['differences_pp'] for r in DATA['development_comparisons'] if r['dataset']==ds and r['method']=='S' and r['reference']=='R') for ds in DS}
    plot.data=[[(i+(seed-1)*.14,diffs[ds][seed]) for i,ds in enumerate(DS)] for seed in range(3)]
    plot.joinedLines=0
    for i,color in enumerate([BLUE,ORANGE,TEAL]):plot.lines[i].symbol=makeMarker('FilledCircle',size=7,fillColor=color,strokeColor=color)
    plot.xValueAxis.valueMin=-.5;plot.xValueAxis.valueMax=2.5;plot.xValueAxis.valueSteps=[0,1,2]
    plot.xValueAxis.labelTextFormat=lambda value:NAMES[int(value)]
    plot.yValueAxis.valueMin=-.4;plot.yValueAxis.valueMax=1.8;plot.yValueAxis.valueStep=.4
    plot.yValueAxis.labelTextFormat=lambda value:f'{value:g}'
    axes(plot);d.add(plot)
    yp=lambda value:180+(value+.4)/2.2*265
    d.add(Line(686,yp(0),1176,yp(0),strokeColor=MUTED,strokeWidth=1.1))
    for i,ds in enumerate(DS):
        mean=sum(diffs[ds])/3;x=686+(i+.5)/3*490
        d.add(Line(x-19,yp(mean),x+19,yp(mean),strokeColor=INK,strokeWidth=2.5))
        label(d,x,126,f'Mean {mean:+.3f}',12,INK,'middle')
    label(d,660,96,'Dots: seed 0 (blue), seed 1 (orange), seed 2 (teal).',12,MUTED)
    label(d,660,79,'Black bars show means; no confidence intervals or causal fit.',12,MUTED)
    label(d,48,63,'S - R means do not replace S - B2: the equal-dataset S - B2 change is only +0.0667 pp.',13,MUTED)
    label(d,48,39,'S and R have identical count multisets in all 9 matched runs; count uniformity alone does not explain the difference.',12,MUTED)
    save(d,'mechanism_S_R')


if __name__=='__main__':
    formal();mechanism();print('Rendered 2 SVG figures with ReportLab chart toolkit.')
