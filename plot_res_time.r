#####################################################################
# plot_res_time.r -- R script to create plots of some of the results
#  these include Figs. S2, S3 and S5: heatmaps of the number of data
#  points and balanced accuracy in each NGA over time, and a barchart
#  of the overall count of each answer for each model

library(arrow)
library(ggplot2)
library(scales)
library(RColorBrewer)

# TODO: set the correct working directory !!
setwd('/home/dkondor/CSH/HoloSim/NLP')


#########################################################
# continuous transformation for the time axis, using a quadratic form
# y_tr = a*y**2 + b*y
# but use a constant factor below 5500 BCE

# we require the following for the derivatives
r1 = 0.1
y0 = -5500
# 2*a*2000 + b = 1
# 2*a*(-5500) + b = 0.1
# -> 15000*a = 0.9 -> 
a = (1-r1)/(4000 - 2*y0)
b = 1-4000*a
# value at -3000
x0 = a*y0^2 + b*y0
# -2365

# breaks and labels used in the plots
breaks1 = c(-10000, -5000, -3000, -2000, -1000, 0, 500, 1000, 1500, 2000)
labels1 = c('10000 BCE', '5000 BCE', '3000 BCE', 
            '2000 BCE', '1000 BCE', '0',
            '500 CE', '1000 CE', '1500 CE', '2000 CE')


# define the transform functions
trfun5 = function(x) {
  x[!is.na(x) & x > y0] = a*x[!is.na(x) & x > y0]^2 + b*x[!is.na(x) & x > y0]
  x[!is.na(x) & x <= y0] = (x[!is.na(x) & x <= y0] - y0)*r1 + x0
  return(-1*x)
}
trinv5 = function(x) {
  x = -1*x
  x[!is.na(x) & x <= x0] = (x[!is.na(x) & x <= x0] - x0) / r1 + y0
  x[!is.na(x) & x > x0] = (-b + sqrt(b*b + 4*a*x[!is.na(x) & x > x0])) / (2*a)
  return(x)
}
tr1 = trans_new("squeeze_quad_lin", trfun5, trinv5)


#################################################################
# Balanced accuracy over time, NGA and region (Fig. S3)
res3 = read_parquet("new_processed_res.parquet")
out_fn_base = "new_region_time_res"

res3 = data.frame(res3)
names(res3)[2] = "year_min"
names(res3)[3] = "year_max"
res3$year_min[res3$year_min < -10000] = -10000

# add line breaks to region names
# unique(res3$region)
res3$region[res3$region == "Central and Southern Asia"] = "Central and\nSouthern Asia"
res3$region[res3$region == "Eastern and South-Eastern Asia"] = "Eastern and\nSouth-Eastern Asia"
res3$region[res3$region == "Northern Africa and Western Asia"] = "Northern Africa,\n Western Asia"
res3$region[res3$region == "Northern America"] = "Northern\nAmerica"
res3$region[res3$region == "Sub-Saharan Africa"] = "Sub-\nSaharan\nAfrica"
res3$region[res3$region == "Latin America and the Caribbean"] = "Latin\nAmerica"

models = unique(res3$model)
# models_df = data.frame(name = models, shortname = c("gpt4o",
#         "gpt35turbo", "llama", "gpt4turbo")) # same order
# new models
models_df = data.frame(name = models, shortname = c("gpt4o", "gpt35turbo",
  "llama-3.1-8B", "llama-3.1-70B", "llama-3-70b", "gpt4turbo", "gemini")) # same order


# create plots for all models (note: only GPT-4-turbo is used)
for(i in 1:nrow(models_df)) {
  res11 = res3[res3$model == models_df$name[i],]
  
  p1 = ggplot(res11) + geom_rect(
    aes(xmin=stage(nga, after_scale = xmin - 0.5),
        xmax=stage(nga, after_scale = xmax + 0.5),
        ymin=year_max, ymax=year_min, fill=balanced_accuracy))
  p1 = p1 + scale_x_discrete()
  p1 = p1 + scale_y_continuous(trans = tr1, breaks = breaks1,
              labels = labels1, limits=c(2001,-10001), expand = c(0,0))
  
  p1 = p1 + theme_bw(6)
  p1 = p1 + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  p1 = p1 + scale_fill_distiller(palette="YlGnBu", limits = c(0.2, 0.66),
                                 direction = 1, oob = squish)
  p1 = p1 + labs(fill="Balanced\naccuracy")
  
  p1 = p1 + facet_grid(~region, scales="free_x", space="free_x")
#  p1 = p1 + ggtitle(models_df$name[i])
  fn1 = paste0(out_fn_base, "_", models_df$shortname[i])
  ggsave(paste0(fn1, ".pdf"), p1, width=6.4, height=4)
  ggsave(paste0(fn1, ".png"), p1, width=6.4, height=4, dpi=300)
}


#################################################################
# Number of data points over time, NGA and region (Fig. S2)
n_data = read_parquet("new_n_data.parquet")
n_data = data.frame(n_data)
names(n_data)[2] = "year_min"
names(n_data)[3] = "year_max"
n_data$year_min[n_data$year_min < -10000] = -10000

n_data$region[n_data$region == "Central and Southern Asia"] = "Central and\nSouthern Asia"
n_data$region[n_data$region == "Eastern and South-Eastern Asia"] = "Eastern and\nSouth-Eastern Asia"
n_data$region[n_data$region == "Northern Africa and Western Asia"] = "Northern Africa,\n Western Asia"
n_data$region[n_data$region == "Northern America"] = "Northern\nAmerica"
n_data$region[n_data$region == "Sub-Saharan Africa"] = "Sub-\nSaharan\nAfrica"
n_data$region[n_data$region == "Latin America and the Caribbean"] = "Latin\nAmerica"

# look manually at the distribution of data
# ggplot(n_data) + geom_histogram(aes(x=N))

p1 = ggplot(n_data) + geom_rect(
  aes(xmin=stage(nga, after_scale = xmin - 0.5),
      xmax=stage(nga, after_scale = xmax + 0.5),
      ymin=year_max, ymax=year_min, fill=N))
p1 = p1 + scale_x_discrete()
p1 = p1 + scale_y_continuous(trans = tr1, breaks = breaks1,
                             labels = labels1, limits=c(2001,-10001), expand = c(0,0))

p1 = p1 + theme_bw(6)
p1 = p1 + theme(axis.text.x = element_text(angle = 45, hjust = 1))
p1 = p1 + scale_fill_distiller(palette="YlGnBu", limits = c(0, 300),
                               direction = 1, oob = squish)
p1 = p1 + labs(fill="Number of\ndata points")

p1 = p1 + facet_grid(~region, scales="free_x", space="free_x")

fn1 = "new_ndata"
ggsave(paste0(fn1, ".pdf"), p1, width=6.4, height=4)
ggsave(paste0(fn1, ".png"), p1, width=6.4, height=4, dpi=300)


####################################################################
# count of answer categories (Fig. S5)
answers = read.csv('new_res_answer_counts.csv')

models = unique(answers$model)
# models_df = data.frame(name = models, shortname = c("GPT-4o", "GPT-3.5",
#     "Llama-3-70b-chat", "GPT-4-turbo", "Seshat")) # same order
# new results
models_df = data.frame(name = models, shortname =  c("GPT-4o", "GPT-3.5-turbo",
              "llama-3.1-8B", "Llama-3.1-70B", "Llama-3-70B",
              "GPT-4-turbo", "Gemini-1.5-flash", "Seshat")) # same order

answers = merge(answers, models_df, by.x="model", by.y="name")
answer_names = data.frame(answer = c('A', 'B', 'C', 'D'),
                          ans_name = c("Present", "Absent", "Inferred\nPresent",
                                       "Inferred\nAbsent"))
answers = merge(answers, answer_names, by="answer")

p1 = ggplot(answers) + geom_col(aes(x=ans_name, y=cnt, fill=shortname),
                                position="dodge")
# colors from colorbrewer
colors = c('#a6cee3','#1f78b4','#b2df8a','#33a02c', '#fb9a99',
          '#e31a1c', '#fdbf6f',  'black')
p1 = p1 + scale_fill_manual(values = colors)
p1 = p1 + theme_bw(6) + xlab("") + ylab("Count") + labs(fill = "Answer source")
ggsave("new_answer_counts.pdf", width=5, height=2.4)



