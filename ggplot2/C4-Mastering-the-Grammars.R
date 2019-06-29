library(ggplot2)

# Building a scatterplot
ggplot(mpg, aes(displ, hwy, colour = factor(cyl))) + geom_point()

ggplot(mpg, aes(displ, hwy, colour = factor(cyl))) + 
  geom_line() + theme(legend.position = "none")

ggplot(mpg, aes(displ, hwy, colour = factor(cyl))) + 
  geom_bar(stat = "identity", position = "identity", fill = NA) + theme(legend.position = "none")

ggplot(mpg, aes(displ, hwy, colour = factor(cyl))) + 
  geom_point() + geom_smooth(method = "lm")

# Adding complexity
ggplot(mpg, aes(displ, hwy)) + geom_point() + geom_smooth() + facet_wrap(~year)