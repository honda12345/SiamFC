# SiamFC-Pytorch
Fully-Convolutional Siamese Networks for Object Tracking  
Pytorch implementation

## Performance
### Benchmark:OTB2015
#### Backbone:Alexnet, Loss:ClassBalancedCELoss
<img src="/reports/OTB2015/SiamFC_defo/precision_plots.png" width=50%><img src="/reports/OTB2015/SiamFC_defo/success_plots.png" width=50%>   
#### Backbone:Alexnet, Loss:FocalLoss
<img src="/reports/OTB2015/SiamFC_focalloss_Alexnet/precision_plots.png" width=50%><img src="/reports/OTB2015/SiamFC_focalloss_Alexnet/success_plots.png" width=50%>  
### Learning Curve
<img src="/reports/learningcurve.png" >   
## Reference
> paper : https://www.robots.ox.ac.uk/~luca/siamese-fc.html  
> referenced code : https://github.com/huanglianghua/siamfc-pytorch  
> Got-10k : http://got-10k.aitestunion.com/  
