
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

csvFile <- "fig-wiki-breakdown.csv"
outfile <- "fig-wiki-breakdown.tex"

options(tikzLatexPackages = 
          c(paste("\\input{",getwd(),"/../packages.tex}\n\\input{",getwd(),"/../macros.tex}",sep="")
          ) 
)

theme_complete_bw <- function(base_size = 12, base_family = "") {
  theme(
    line =               element_line(colour = "black", size = 0.5, linetype = 1,lineend = "butt"),
    rect =               element_rect(fill = "white", colour = "black", size = 0.5, linetype = 1),
    text =               element_text(family = base_family, face = "plain",colour = "black", size = base_size,hjust = 0.5, vjust = 0.5, angle = 0, lineheight = 0.9),
    axis.text =          element_text(size = rel(0.6), colour = "black"),
    axis.title.y =       element_text(size = rel(0.7),angle = 90, colour = "black",vjust = 0.9),
    axis.title.x =       element_text(size = rel(0.7), colour = "black"),
    strip.text =         element_text(size = rel(0.6)),
    legend.background =  element_blank(),
    legend.margin =      unit(0.0, "cm"),
    legend.key =         element_blank(),
    legend.key.height =  unit(0.2, "cm"),
    legend.key.width =   unit(0.2, "cm"),
    legend.text =        element_text(size = rel(0.4)),
    legend.text.align =  0,
    legend.title =       element_blank(),
    legend.title.align = NULL,
    legend.justification = "center",
    legend.direction = "vertical",
    legend.box =         NULL,
    legend.position = c(0.15,0.87),
    panel.background =   element_rect(fill = NA, colour = "grey", size = 1.3),
    panel.border =       element_blank(),
    panel.grid.major =   element_line(colour = "grey90", size = 0.7),
    panel.grid.minor =   element_line(colour = "grey90", size = 0.3),
    panel.margin =       unit(0.7, "lines"),
    strip.background =   element_rect(fill = NA, colour = NA),
    strip.text.x =       element_text(colour = "black", size = base_size * 0.6),
    strip.text.y =       element_text(colour = "black", size = base_size * 0.6, angle = -90),
    plot.background =    element_rect(colour = NA, fill = "white"),
    plot.title =         element_text(size = base_size * 1.2),
    plot.margin=         unit(c(1,0,0,0),"mm"),
    complete = TRUE
  )
}

mf_labeller <- function(var, value){
    value <- as.character(value)
    if (var=="index") { 
        value[value=="single"] <- "\\singleCST"
        value[value=="dual"]   <- "\\dualCST"
    }
    return(value)
}

d <- read.csv(csvFile,sep=";")
d$time_ms <- d$time * 1000
d$time_ms_per_sentence <- d$time_ms / 10000
d$func <- factor(d$func, levels = c("N1PlusFrontBack","forward_search","backward_search","N1PlusBack","N1PlusFront")
	, labels=c("\\nlplusfrontbackname","\\forwardsearchname","\\backwardsearchname","\\nlplusbackname","\\nlplusfrontname"),ordered =TRUE)

plot <- ggplot(d,aes(time_ms_per_sentence,x=factor(ngram),fill=func,order=func))
plot <- plot + geom_bar(stat="identity")
plot <- plot + facet_grid(index ~ . ,scales="free_y",labeller=mf_labeller)
plot <- plot + scale_fill_brewer(palette="Set1")
plot <- plot + theme_complete_bw()
plot <- plot + scale_y_continuous(name="Time per Sentence [msec]")
plot <- plot + scale_x_discrete(name="\\ngram size",labels=c(2,3,4,5,6,8,10,"$\\infty$"))
#plot <- plot + scale_x_log10(name="Space Usage [bytes]",labels=f2si)
#plot <- plot + scale_y_log10(name="Time [sec]")

tikz(outfile,width = 3, height = 2)
print(plot)
dev.off();
