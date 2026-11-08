"""Scientific trace figure using ReportLab; all data are saved, audited records."""
from pathlib import Path
import csv
import importlib.util
import math
import statistics as st
from reportlab.graphics.shapes import Drawing,Rect,Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.colors import white,HexColor

ROOT=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('small_figure_style',ROOT/'tools/render_small_figures.py')
style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)


def main():
    with (ROOT/'tables/arxiv_A_epochs_60.csv').open(encoding='utf-8-sig') as stream:epochs=list(csv.DictReader(stream))
    with (ROOT/'tables/arxiv_A_validation_curves_10000.csv').open(encoding='utf-8-sig') as stream:validation=list(csv.DictReader(stream))
    curves={epoch:[] for epoch in range(1000)}
    for row in validation:curves[int(row['epoch'])].append(float(row['validation_accuracy_percent']))
    assert len(epochs)==60 and all(len(v)==10 for v in curves.values())
    d=Drawing(1250,620);d.add(Rect(0,0,1250,620,fillColor=white,strokeColor=None))
    style.label(d,48,575,'ogbn-arxiv: one B0 encoder and ten fixed linear probes',25,bold=True)
    style.label(d,48,547,'Complete saved traces; resource-adapted batch 256. Validation curves are not used to select pretraining epochs.',13,style.MUTED)
    style.label(d,70,505,'A. Complete pretraining trajectory',17,bold=True)
    left=LinePlot();left.x=78;left.y=160;left.width=480;left.height=300
    left.data=[[(int(row['epoch']),float(row['loss_mean'])) for row in epochs]];left.joinedLines=1
    left.lines[0].strokeColor=style.BLUE;left.lines[0].strokeWidth=2.2
    left.xValueAxis.valueMin=1;left.xValueAxis.valueMax=60;left.xValueAxis.valueSteps=[1,10,20,30,40,50,60]
    left.yValueAxis.valueMin=.5;left.yValueAxis.valueMax=3.;left.yValueAxis.valueSteps=[.5,1,1.5,2,2.5,3]
    left.yValueAxis.labelTextFormat=lambda value:f'{value:g}';style.axes(left);d.add(left)
    x=78+(41-1)/59*480;d.add(Line(x,160,x,460,strokeColor=style.ORANGE,strokeDashArray=[4,3]))
    style.label(d,78,481,'Mean training loss (fixed original objective)',12,style.MUTED)
    style.label(d,318,115,'Pretraining epoch (1-60)',12,style.MUTED,'middle')
    style.label(d,78,85,'Dashed line: EMA active from epoch 41 (zero-based argument 40).',11,style.MUTED)
    style.label(d,78,65,'This marker does not identify a causal effect of EMA.',11,style.MUTED)
    style.label(d,665,505,'B. Linear-probe validation trajectories',17,bold=True)
    right=LinePlot();right.x=690;right.y=160;right.width=480;right.height=300
    right.data=[[(epoch,min(curves[epoch])) for epoch in curves],[(epoch,max(curves[epoch])) for epoch in curves],
                [(epoch,st.mean(curves[epoch])) for epoch in curves]];right.joinedLines=1
    for i in [0,1]:right.lines[i].strokeColor=HexColor('#b8c2cc');right.lines[i].strokeWidth=.8
    right.lines[2].strokeColor=style.BLUE;right.lines[2].strokeWidth=1.8
    right.xValueAxis.valueMin=0;right.xValueAxis.valueMax=999;right.xValueAxis.valueSteps=[0,200,400,600,800,999]
    minimum=math.floor(min(min(v) for v in curves.values())/5)*5
    right.yValueAxis.valueMin=minimum;right.yValueAxis.valueMax=75;right.yValueAxis.valueSteps=list(range(minimum,76,5))
    right.yValueAxis.labelTextFormat=lambda value:f'{value:.0f}';style.axes(right);d.add(right)
    style.label(d,690,481,'Validation accuracy (%)',12,style.MUTED)
    style.label(d,930,115,'Classifier epoch (0-999)',12,style.MUTED,'middle')
    style.label(d,690,85,'Blue: mean; gray: min/max across ten classifiers of one encoder.',11,style.MUTED)
    style.label(d,690,65,'Min/max lines are not confidence intervals; no test curve is shown.',11,style.MUTED)
    style.label(d,48,30,'Sources: 60 complete pretraining epoch summaries and 10,000 saved classifier validation records.',12,style.MUTED)
    style.save(d,'arxiv_A_training_and_probe')
    print('Rendered arxiv trace SVG without fitting or additional experiments.')


if __name__=='__main__':main()
