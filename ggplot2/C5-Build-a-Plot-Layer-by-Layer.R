library(ggplot2)
library(dplyr)

# Build a plot
p <- ggplot(mpg, aes(displ, hwy))
p + geom_point()

# Data
mod <- loess(hwy ~ displ, data = mpg)
grid <- data_frame(displ = seq(min(mpg$displ), max(mpg$displ), length = 50))
grid$hwy <- predict(mod, newdata = grid)
grid

std_resid <- resid(mod) / mod$s
outlier <- filter(mpg, abs(std_resid) > 2)
outlier

ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  geom_line(data = grid, colour = "blue", size = 1.5) +
  geom_text(data = outlier, aes(label = model))

# Aesthetic Mappings
## Specifying the Aesthetics in the Plot vs in the Layers
ggplot(mpg, aes(displ, hwy, colour = class)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  theme(legend.position = "none")

ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(colour = class)) +
  geom_smooth(method = "lm", se = FALSE) +
  theme(legend.position = "none")

## Setting vs Mapping
ggplot(mpg, aes(cty, hwy)) +
  geom_point(colour = "darkblue")
ggplot(mpg, aes(cty, hwy)) +
  geom_point(aes(colour = "darkblue"))
ggplot(mpg, aes(cty, hwy)) +
  geom_point(aes(colour = "darkblue")) + scale_colour_identity()
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  geom_smooth(aes(colour = "loess"), method = "loess", se = FALSE) +
  geom_smooth(aes(colour = "lm"), method = "lm", se = FALSE) +
  labs(colour = "Method")

# Stats
ggplot(mpg, aes(trans, cty)) + geom_point() +
  stat_summary(geom = "point", fun.y = "mean", colour = "red", size = 4)
ggplot(mpg, aes(trans, cty)) + geom_point() +
  stat_summary(stat = "summary", fun.y = "mean", colour = "red", size = 4)

## Generated Variables
ggplot(diamonds, aes(price)) +
  geom_histogram(binwidth = 500)

ggplot(diamonds, aes(price)) +
  geom_histogram(aes(y = ..density..), binwidth = 500)

ggplot(diamonds, aes(price, colour = cut)) +
  geom_freqpoly(binwidth = 500) +
  theme(legend.position = "none")

ggplot(diamonds, aes(price, colour = cut)) +
  geom_freqpoly(aes(y = ..density..), binwidth = 500) +
  theme(legend.position = "none")

# Position Adjustments
dplot <- ggplot(diamonds, aes(color, fill = cut)) +
  xlab(NULL) + ylab(NULL) + theme(legend.position = "none")
dplot + geom_bar()
dplot + geom_bar(position = "fill")
dplot + geom_bar(position = "dodge")
dplot + geom_bar(position = "identity", alpha = 1 / 2, colour = "grey50")

ggplot(diamonds, aes(color, colour = cut)) +
  geom_line(aes(group = cut), stat = "count") +
  xlab(NULL) + ylab(NULL) + theme(legend.position = "none")

ggplot(mpg, aes(displ, hwy)) +
  geom_jitter(width = 0.05, height = 0.5)