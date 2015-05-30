
require("ggplot2")
require("scales")
require("grid")
require("gridExtra")
require("MASS")
require("scales")
require("sitools")
library(tikzDevice)
library(stringr)
library(plyr)
latex_percent <- function (x) {
    x <- plyr::round_any(x, scales:::precision(x)/100)
    stringr::str_c(comma(x * 100), "\\%")
}

csvFile <- "fig-eu-de-space-time.csv"
outfile <- "fig-eu-de-space-time.tex"

# options(tikzLatexPackages = 
#           c(paste("\\input{",getwd(),"/../packages.tex}\n\\input{",getwd(),"/../macros.tex}",sep="")
#           ) 
# )

theme_complete_bw <- function(base_size = 12, base_family = "") {
  theme(
    line =               element_line(colour = "black", size = 0.5, linetype = 1,lineend = "butt"),
    rect =               element_rect(fill = "white", colour = "black", size = 0.5, linetype = 1),
    text =               element_text(family = base_family, face = "plain",colour = "black", size = base_size,hjust = 0.5, vjust = 0.5, angle = 0, lineheight = 0.9),
    axis.text =          element_text(size = rel(0.6), colour = "black"),
    axis.title.y =       element_text(size = rel(0.7),angle = 90, colour = "black",vjust = 0.2),
    axis.title.x =       element_text(size = rel(0.7), colour = "black"),
    strip.text =         element_text(size = rel(0.6)),
    legend.background =  element_blank(),
    legend.margin =      unit(0.0, "cm"),
    legend.key =         element_blank(),
    legend.key.height =  unit(0.3, "cm"),
    legend.key.width =   unit(0.3, "cm"),
    legend.text =        element_text(size = rel(0.5)),
    legend.text.align =  0,
    legend.title =       element_blank(),
    legend.title.align = NULL,
    legend.justification = "center",
    legend.direction = "horizontal",
    legend.box =         NULL,
    legend.position = c(0.5,-0.3),
    panel.background =   element_rect(fill = NA, colour = "grey", size = 1.3),
    panel.border =       element_blank(),
    panel.grid.major =   element_line(colour = "grey90", size = 0.7),
    panel.grid.minor =   element_line(colour = "grey90", size = 0.3),
    panel.margin =       unit(0.5, "lines"),
    strip.background =   element_rect(fill = NA, colour = NA),
    strip.text.x =       element_text(colour = "black", size = base_size * 0.6),
    strip.text.y =       element_text(colour = "black", size = base_size * 0.6, angle = -90),
    plot.background =    element_rect(colour = NA, fill = "white"),
    plot.title =         element_text(size = base_size * 1.2),
    plot.margin=         unit(c(0,1,3.5,-1),"mm"),
    complete = TRUE
  )
}

mf_labeller <- function(var, value){
    value <- as.character(value)
    if (var=="process") { 
        value[value=="construction"] <- "Construction Cost"
        value[value=="query"]   <- "Query Cost"
    }
    return(value)
}

d <- read.csv(csvFile,sep=";")
d$space <- d$space * 1024*1024
d$timepersentence <- d$time / 10000

srlmdef <- subset(d,d$method %in% c("srilm-default"))
srlmc <- subset(d,d$method %in% c("srilm-compact"))
dualcst <- subset(d,d$method %in% c("dual-CST"))
dualcsttwo <- subset(dualcst,dualcst$ngram == 2)
dualcstrest <- subset(dualcst,dualcst$ngram != 2)
singlecst <- subset(d,d$method %in% c("single-CST"))
singlecsttwo <- subset(singlecst,singlecst$ngram == 2)
singlecstrest <- subset(singlecst,singlecst$ngram != 2)

plot <- ggplot(d,aes(type,y=time,x=space,group=factor(method),colour=factor(method),shape=factor(method)))
plot <- plot + annotation_logticks(sides = "lb",short = unit(.5,"mm"), mid = unit(1,"mm"), long = unit(1.7,"mm"))
plot <- plot + geom_point(size=1.5)
plot <- plot + facet_grid(~ process,labeller=mf_labeller)
plot <- plot + scale_x_log10(name="Space Usage [bytes]",labels=f2si)
plot <- plot + scale_y_log10(breaks=c(10,100,1000,10000),name="Time [sec]",labels=f2si)
plot <- plot + theme_complete_bw()
#plot <- plot + geom_text(size=1.5,hjust=1.7,aes(x=space, y=time, label=lab, group=NULL),data=subset(d,d$method %in% c("dual-CST")))
#plot <- plot + geom_text(size=1.5,vjust=-1.2,aes(x=space, y=time, label=lab, group=NULL),data=subset(d,d$method %in% c("dual-CST","srilm-compact")))
plot <- plot + geom_text(size=1.5,hjust=-0.7,vjust=1.7,aes(x=space, y=time, label=lab, group=NULL),data=subset(srlmdef,srlmdef$process=="construction"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,vjust=-1.2,aes(x=space, y=time, label=lab, group=NULL),data=subset(srlmc,srlmc$process=="construction"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,hjust=1,vjust=-1.2,aes(x=space, y=time, label=lab, group=NULL),data=subset(dualcst,dualcst$process=="construction"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,vjust=1.2,aes(x=space, y=time, label=lab, group=NULL),data=subset(singlecst,singlecst$process=="construction"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,hjust=-1,aes(x=space, y=time, label=lab, group=NULL),data=subset(dualcsttwo,dualcsttwo$process=="query"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,hjust=-0.4,aes(x=space, y=time, label=lab, group=NULL),data=subset(dualcstrest,dualcstrest$process=="query"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,hjust=-1.1,aes(x=space, y=time, label=lab, group=NULL),data=subset(singlecsttwo,singlecsttwo$process=="query"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,hjust=-0.4,aes(x=space, y=time, label=lab, group=NULL),data=subset(singlecstrest,singlecstrest$process=="query"),show_guide  = FALSE)

plot <- plot + geom_text(size=1.5,vjust=-1.4,aes(x=space, y=time, label=lab, group=NULL),data=subset(srlmdef,srlmdef$process=="query"),show_guide  = FALSE)
plot <- plot + geom_text(size=1.5,vjust=1.8,aes(x=space, y=time, label=lab, group=NULL),data=subset(srlmc,srlmc$process=="query"),show_guide  = FALSE)
plot <- plot+ guides(colour = guide_legend(ncol = 4))
#plot <- plot + geom_text(size=1.5,vjust=2,aes(x=space, y=time, label=lab, group=NULL),data=subset(d,d$method %in% c("single-CST")))

tikz(outfile,width = 3, height = 2)
print(plot)
dev.off();
