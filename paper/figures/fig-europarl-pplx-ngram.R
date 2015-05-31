
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

csvFile <- "fig-europarl-pplx-ngram.csv"
outfile <- "fig-europarl-pplx-ngram.tex"

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
    strip.text =         element_text(size = rel(0.7)),
    legend.background =  element_blank(),
    legend.margin =      unit(0.0, "cm"),
    legend.key =         element_blank(),
    legend.key.height =  unit(0.25, "cm"),
    legend.key.width =   unit(0.55, "cm"),
    legend.text =        element_text(size = rel(0.5)),
    legend.text.align =  0,
    legend.title =       element_text(size = rel(0.6)),
    legend.title.align = NULL,
    legend.justification = "center",
    legend.direction = "vertical",
    legend.box =         NULL,
    legend.position = c(0.6,0.65),
    panel.background =   element_rect(fill = NA, colour = "grey", size = 1.3),
    panel.border =       element_blank(),
    panel.grid.major =   element_line(colour = "grey90", size = 0.7),
    panel.grid.minor =   element_line(colour = "grey90", size = 0.3),
    panel.margin =       unit(0.7, "lines"),
    strip.background =   element_rect(fill = NA, colour = NA),
    strip.text.x =       element_text(colour = "black", size = base_size * 0.8),
    strip.text.y =       element_text(colour = "black", size = base_size * 0.8, angle = -90),
    plot.background =    element_rect(colour = NA, fill = "white"),
    plot.title =         element_text(size = base_size * 1.2),
    plot.margin=         unit(c(1,0,0,0),"mm"),
    complete = TRUE
  )
}

d <- read.csv(csvFile,sep=";")
d$relpplx <- (100 * d$pplx / d$inf) - 99

d$language <- factor(d$language,levels=c("BG","CZ","DE","EN","FI","FR","HU","IT","PT"),
    labels=c("BG ($2$-gram $117.39$, $\\infty$-gram $73.01$)","CZ ($2$-gram $232.36$, $\\infty$-gram $161.15$)","DE ($2$-gram $178.11$, $\\infty$-gram $108.27$)",
    "EN ($2$-gram $67.14$, $\\infty$-gram $59.92$)","FI ($2$-gram $446.29$, $\\infty 314.22$)","FR ($2$-gram $89.95$, $\\infty$-gram $47.91$)",
    "HU ($2$-gram $251.49$, $\\infty$-gram $182.18$)","IT ($2$-gram $132.26$, $\\infty$-gram $77.80$)","PT ($2$-gram $121.42$, $\\infty$-gram $68.58$)" ))

plot <- ggplot(d,aes(type,y=relpplx,x=factor(ngram),group=factor(language),linetype=factor(language),colour=factor(language)))
plot <- plot + geom_line(size=1)
plot <- plot + geom_point(size=1)
plot <- plot + theme_complete_bw()
plot <- plot + scale_y_log10(name="Perplexity [Relative to $\\infty$]",
breaks=c(1,2,5,10,20,50,100),labels=c("100\\%","102\\%","105\\%","110\\%","120\\%","150\\%","200\\%")
	)
plot <- plot + scale_x_discrete(name="\\ngram size",labels=c("2","3","4","5","6","7","8","9","10","15","20","$\\infty$"))
plot <- plot+ guides(colour = guide_legend(ncol = 1))
plot <- plot + scale_colour_brewer(palette="Set1",name="Language ($2$-gram pplx, $\\infty$-gram pplx)")
plot <- plot + scale_linetype_discrete(name="Language ($2$-gram pplx, $\\infty$-gram pplx)")
plot <- plot + annotation_logticks(sides = "l")


tikz(outfile,width = 3, height = 2)
print(plot)
dev.off();
