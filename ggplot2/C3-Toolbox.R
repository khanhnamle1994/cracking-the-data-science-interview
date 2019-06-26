library(ggplot2)

# Basic Plot Types
df <- data.frame(
  x = c(3, 1, 5),
  y = c(2, 4, 6),
  label = c("a", "b", "c")
)

p <- ggplot(df, aes(x, y, label = label)) +
  labs(x = NULL, y = NULL) +
  theme(plot.title = element_text(size = 12))

p + geom_point() + ggtitle("point")
p + geom_text() + ggtitle("text")
p + geom_bar(stat = "identity") + ggtitle("bar")
p + geom_tile() + ggtitle("raster")
p + geom_line() + ggtitle("line")
p + geom_area() + ggtitle("area")
p + geom_path() + ggtitle("path")
p + geom_polygon() + ggtitle("polygon")

# Labels
df <- data.frame(x = 1, y = 3:1, family = c("sans", "serif", "mono"))
ggplot(df, aes(x, y)) + geom_text(aes(label = family, family = family))

df <- data.frame(
  x = c(1, 1, 2, 2, 1.5),
  y = c(1, 2, 1, 2, 1.5),
  text = c("bottom-left", "bottom-right", "top-left", "top-right", "center")
)
ggplot(df, aes(x, y)) + geom_text(aes(label = text))
ggplot(df, aes(x, y)) + geom_text(aes(label = text), vjust = "inward", hjust = "inward")

df <- data.frame(trt = c("a", "b", "c"), resp = c(1.2, 3.4, 2.5))
ggplot(df, aes(resp, trt)) +
  geom_point() +
  geom_text(aes(label = paste0("(", resp, ")")), nudge_y = -0.25) + xlim(1, 3.6)

ggplot(mpg, aes(displ, hwy)) + geom_text(aes(label = model)) + xlim(1, 8)
ggplot(mpg, aes(displ, hwy)) + geom_text(aes(label = model), check_overlap = TRUE) + xlim(1, 8)

label <- data.frame(
  waiting = c(55, 80),
  eruptions = c(2, 4.3),
  label = c("peak one", "peak two")
)

ggplot(faithfuld, aes(waiting, eruptions)) + 
  geom_tile(aes(full = density)) +
  geom_label(data = label, aes(label = label))

ggplot(mpg, aes(displ, hwy, colour = class)) + geom_point()

ggplot(mpg, aes(displ, hwy, colour = class)) + 
  geom_point(show.legend = FALSE) + 
  directlabels::geom_dl(aes(label = class), method = "smart.grid")

# Annotations
ggplot(economics, aes(date, unemploy)) + geom_line()

presidential <- subset(presidential, start > economics$date[1])
ggplot(economics) + 
  geom_rect(
    aes(xmin = start, xmax = end, fill = party),
    ymin = -Inf, ymax = Inf, alpha = 0.2,
    data = presidential
  ) + 
  geom_vline(
    aes(xintercept = as.numeric(start)),
    data = presidential,
    colour = "grey50", alpha = 0.5
  ) + 
  geom_text(
    aes(x = start, y = 2500, label = name),
    data = presidential,
    size = 3, vjust = 0, hjust = 0, nudge_x = 50
  ) + 
  geom_line(aes(date, unemploy)) +
  scale_fill_manual(values = c("blue", "red"))

yrng <- range(economics$unemploy)
xrng <- range(economics$date)
caption <- paste(strwrap("Unemployment rates in the US have 
                         varied a lot over the years", 40), collapse = "\n")
ggplot(economics, aes(date, unemploy)) + 
  geom_line() + 
  geom_text(
    aes(x, y, label = caption),
    data = data.frame(x = xrng[1], y = yrng[2], caption = caption),
    hjust = 0, vjust = 1, size = 4
  )

ggplot(diamonds, aes(log10(carat), log10(price))) +
  geom_bin2d() + facet_wrap(~cut, nrow = 1)

mod_coef <- coef(lm(log10(price) ~ log10(carat), data = diamonds))
ggplot(diamonds, aes(log10(carat), log10(price))) +
  geom_bin2d() + 
  geom_abline(intercept = mod_coef[1], slope = mod_coef[2], colour = "white", size = 1) +
  facet_wrap(~cut, nrow = 1)

# Collective Geoms
data(Oxboys, package = "nlme")
head(Oxboys)

## Multiple Groups, One Aesthetic
ggplot(Oxboys, aes(age, height, group = Subject)) + geom_point() + geom_line()
ggplot(Oxboys, aes(age, height)) + geom_point() + geom_line()

## Different Groups on Different Layers
ggplot(Oxboys, aes(age, height, group = Subject)) + geom_line() +
  geom_smooth(method = "lm", se = FALSE)
ggplot(Oxboys, aes(age, height)) + geom_line(aes(group = Subject)) +
  geom_smooth(method = "lm", size = 2, se = FALSE)

## Overriding the Default Grouping
ggplot(Oxboys, aes(Occasion, height)) + geom_boxplot()
ggplot(Oxboys, aes(Occasion, height)) + geom_boxplot() +
  geom_line(aes(group = Subject), colour = "#3366FF", alpha = 0.5)

## Matching Aesthetics to Graphic Objects
df <- data.frame(x = 1:3, y = 1:3, colour = c(1, 3, 5))
ggplot(df, aes(x, y, colour = factor(colour))) +
  geom_line(aes(group = 1), size = 2) +
  geom_point(size = 5)
ggplot(df, aes(x, y, colour = colour)) +
  geom_line(aes(group = 1), size = 2) +
  geom_point(size = 5)

xgrid <- with(df, seq(min(x), max(x), length = 50))
interp <- data.frame(
  x = xgrid, 
  y = approx(df$x, df$y, xout = xgrid)$y,
  colour = approx(df$x, df$colour, xout = xgrid)$y
)
ggplot(interp, aes(x, y, colour = colour)) +
  geom_line(size = 2) + geom_point(data = df, size = 5)

ggplot(mpg, aes(class)) + geom_bar()
ggplot(mpg, aes(class, fill = drv)) + geom_bar()

ggplot(mpg, aes(class, fill = hwy)) + geom_bar()
ggplot(mpg, aes(class, fill = hwy, group = hwy)) + geom_bar()

# Surface Plots
ggplot(faithfuld, aes(eruptions, waiting)) + geom_contour(aes(z = density, colour = ..level..))
ggplot(faithfuld, aes(eruptions, waiting)) + geom_raster(aes(fill = density))

small <- faithfuld[seq(1, nrow(faithfuld), by = 10), ]
ggplot(small, aes(eruptions, waiting)) +
  geom_point(aes(size = density), alpha = 1/3) + scale_size_area()

# Revealing Uncertainty
y <- c(18, 11, 16)
df <- data.frame(x = 1:3, y = y, se = c(1.2, 0.5, 1.0))
base <- ggplot(df, aes(x, y, ymin = y - se, ymax = y + se))
base + geom_crossbar()
base + geom_pointrange()
base + geom_smooth(stat = "identity")
base + geom_errorbar()
base + geom_linerange()
base + geom_ribbon()

# Displaying Distributions over Diamond Data
diamonds
ggplot(diamonds, aes(depth)) + geom_histogram()
ggplot(diamonds, aes(depth)) + geom_histogram(binwidth = 0.1) + xlim(55, 70)

ggplot(diamonds, aes(depth)) +
  geom_freqpoly(aes(colour = cut), binwidth = 0.1, na.rm = TRUE) +
  xlim(58, 68) + theme(legend.position = "none")
ggplot(diamonds, aes(depth)) +
  geom_histogram(aes(fill = cut), binwidth = 0.1, position = "fill", na.rm = TRUE) +
  xlim(58, 68) + theme(legend.position = "none")
ggplot(diamonds, aes(depth)) +
  geom_density(na.rm = TRUE) +
  xlim(58, 68) + theme(legend.position = "none")
ggplot(diamonds, aes(depth, fill = cut, colour = cut)) +
  geom_density(alpha = 0.2, na.rm = TRUE) +
  xlim(58, 68) + theme(legend.position = "none")

# Dealing with Overplotting
df <- data.frame(x = rnorm(2000), y = rnorm(2000))
norm <- ggplot(df, aes(x, y)) + xlab(NULL) + ylab(NULL)
norm + geom_point()
norm + geom_point(shape = 1) # Hollow circles
norm + geom_point(shape = ".") # Pixel sized
norm + geom_point(alpha = 1 / 3)
norm + geom_point(alpha = 1 / 5)
norm + geom_point(alpha = 1 / 10)
norm + geom_bin2d()
norm + geom_bin2d(bins = 10)
norm + geom_hex()
norm + geom_hex(bins = 10)

# Statistical Summaries
ggplot(diamonds, aes(color)) + geom_bar()
ggplot(diamonds, aes(color, price)) + geom_bar(stat = "summary_bin", fun.y = mean)
ggplot(diamonds, aes(table, depth)) + geom_bin2d(binwidth = 1, na.rm = TRUE) +
  xlim(50, 70) + ylim(50, 70)
ggplot(diamonds, aes(table, depth, z = price)) + 
  geom_raster(binwidth = 1, stat = "summary_2d", fun = mean, na.rm = TRUE) +
  xlim(50, 70) + ylim(50, 70)