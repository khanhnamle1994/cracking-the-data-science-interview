library(ggplot2)
# Fuel economy data
mpg
# Key components - data, aesthetic mappings, and geoms
ggplot(mpg, aes(x = displ, y = cty, colour = class)) + geom_point()
# Color, size, shape, and other aesthetic attributes
ggplot(mpg, aes(displ, hwy)) + geom_point(aes(colour = "blue"))
ggplot(mpg, aes(displ, hwy)) + geom_point(colour = "blue")
# Facetting
ggplot(mpg, aes(displ, hwy)) + geom_point() + facet_wrap(~class)

# Plot geoms
## Adding a smoother to a plot
ggplot(mpg, aes(displ, hwy)) + geom_point() + geom_smooth()
ggplot(mpg, aes(displ, hwy)) + geom_point() + geom_smooth(span=0.2)
ggplot(mpg, aes(displ, hwy)) + geom_point() + geom_smooth(span=1)

library(mgcv)
ggplot(mpg, aes(displ, hwy)) + geom_point() + geom_smooth(method = "gam", formula = y ~ s(x))
ggplot(mpg, aes(displ, hwy)) + geom_point() + geom_smooth(method = "lm")

## Boxplots and jittered points
ggplot(mpg, aes(drv, hwy)) + geom_point()
ggplot(mpg, aes(drv, hwy)) + geom_jitter()
ggplot(mpg, aes(drv, hwy)) + geom_boxplot()
ggplot(mpg, aes(drv, hwy)) + geom_violin()

## Histograms and frequency polygons
ggplot(mpg, aes(hwy)) + geom_histogram()
ggplot(mpg, aes(hwy)) + geom_freqpoly(binwidth=2.5)
ggplot(mpg, aes(hwy)) + geom_freqpoly(binwidth=1)

ggplot(mpg, aes(displ, colour = drv)) + geom_freqpoly(binwidth=0.5)
ggplot(mpg, aes(displ, colour = drv)) + geom_histogram(binwidth = 0.5) + facet_wrap(~drv, ncol = 1)

## Bar charts
ggplot(mpg, aes(manufacturer)) + geom_bar()

drugs <- data.frame(
  drug = c("a", "b", "c"),
  effect = c(4.2, 9.7, 6.1)
)

ggplot(drugs, aes(drug, effect)) + geom_bar(stat = "identity")
ggplot(drugs, aes(drug, effect)) + geom_point()

## Time series with line and path plots
ggplot(economics, aes(date, unemploy / pop)) + geom_line()
ggplot(economics, aes(date, uempmed)) + geom_line()

ggplot(economics, aes(unemploy / pop, uempmed)) + geom_path() + geom_point()
year <- function(x) as.POSIXlt(x)$year + 1900
ggplot(economics, aes(unemploy / pop, uempmed)) + 
  geom_path(colour = "grey50") + geom_point(aes(colour = year(date)))

# Modifying the axes
ggplot(mpg, aes(cty, hwy)) + geom_point(alpha = 1/3)
ggplot(mpg, aes(cty, hwy)) + geom_point(alpha = 1/3) + 
  xlab("city driving (mpg)") + ylab("highway driving (mpg")
ggplot(mpg, aes(cty, hwy)) + geom_point(alpha = 1/3) + 
  xlab(NULL) + ylab(NULL)

ggplot(mpg, aes(drv, hwy)) + geom_jitter(width = 0.25)
ggplot(mpg, aes(drv, hwy)) + geom_jitter(width = 0.25) +
  xlim("f", "r") + ylim(20, 30)
ggplot(mpg, aes(drv, hwy)) + geom_jitter(width = 0.25) +
  geom_jitter(width = 0.25, na.rm = TRUE) + ylim(NA, 30)

# Output
p <- ggplot(mpg, aes(displ, hwy, colour = factor(cyl))) + geom_point()
print(p)
summary(p)

# Quick plots
qplot(displ, hwy, data = mpg)
qplot(displ, data = mpg)

qplot(displ, hwy, data = mpg, colour = "blue")
qplot(displ, hwy, data = mpg, colour = I("blue"))