library("rjson")
library(ggplot2)
library("gg3D")
library(dplyr)

result_dir = "C:/Users/julia/Downloads/results"
output_dir = "C:/Users/julia/Downloads/viz"

res_paths = Sys.glob(paths = "C:/Users/julia/Downloads/results/*/*/*")

ev_res <- lapply(res_paths, FUN = function(path_oi){
  return(rjson::fromJSON(paste(readLines(paste(path_oi, "eval", "result.txt", sep = "/")))))
})

# vergeleichbarkeit
perc_tracked = sapply(ev_res, FUN = function(list_oi){
  return(max(list_oi$traj_timestamp)/list_oi$seq_len*100)
})

algos = sapply(ev_res, FUN = function(list_oi){
  return(list_oi$algo)
})

sequences = sapply(ev_res, FUN = function(list_oi){
  return(list_oi$sequence)
})

runs = sapply(ev_res, FUN = function(list_oi){
  return(list_oi$run)
})

fps = sapply(ev_res, FUN = function(list_oi){
  return(list_oi$fps)
})

ttime = sapply(ev_res, FUN = function(list_oi){
  return(round(list_oi$proc_time,2))
})

mean_dist = sapply(ev_res, FUN = function(list_oi){
  return(mean(list_oi$traj_dist))
})

mean_dist_pc = sapply(ev_res, FUN = function(list_oi){
  return(mean(list_oi$pq_errors))
})


g <- ggplot(data=perc_df, aes(seq, perc_tracked, fill=algorithm)) + geom_boxplot()
g <- g + xlab(label="Sequence") + ylab("Percentage of the Sequence, which was tracked by the algorithm")
ggsave(filename = "perc_seq.png", plot = g, path = "C:/Users/julia/Downloads", 
       width = 11, height = 6)

perc_df <- data.frame(
  "algorithm" = algos, 
  "seq" = sequences, 
  "run" = runs, 
  "perc_tracked" = perc_tracked
)

pq_error_df <- data.frame(
  "algorithm" = algos, 
  "seq" = sequences, 
  "run" = runs, 
  "pc_error" = mean_dist_pc
)

mean_dist_df <- data.frame(
  "algorithm" = algos, 
  "seq" = sequences, 
  "run" = runs, 
  "distance" = mean_dist
)

fps_df <- data.frame(
  "algorithm" = algos, 
  "seq" = sequences, 
  "run" = runs, 
  "fps" = fps,
  "proc_time" = ttime
)

g <- ggplot(data=mean_dist_df, aes(seq, distance, fill=algorithm)) + geom_boxplot()
g <- g + xlab(label="Sequence") + ylab("Distance ground truth point in m")
ggsave(filename = "dist_error.png", plot = g, path = "C:/Users/julia/Downloads", 
       width = 11, height = 6)




# pointcloud
pc_errors = sapply(ev_res, FUN = function(list_oi){
  return(mean(list_oi$pq_errors))
})

algos = sapply(ev_res, FUN = function(list_oi){
  return(list_oi$algo)
})

sequences = sapply(ev_res, FUN = function(list_oi){
  return(list_oi$sequence)
})

runs = sapply(ev_res, FUN = function(list_oi){
  return(list_oi$run)
})

pc_df <- data.frame(
  "algorithm" = algos, 
  "seq" = sequences, 
  "run" = runs, 
  "pc_error" = pc_errors
)

pc_df_plot <- pc_df[!is.na(pc_df$pc_error),]

g <- ggplot(data=pc_df_plot, aes(seq, pc_error, fill=algorithm)) + geom_boxplot()
g <- g + xlab(label="Sequence") + ylab("Distance to closest ground truth point in m")
ggsave(filename = "pc_error.png", plot = g, path = "C:/Users/julia/Downloads", 
       width = 11, height = 6)

pc_errors_ORB <- data.frame(
  "algo" = "ORB", 
  "error" = ev_res[[52]]$pq_errors
)

pc_errors_DSO <- data.frame(
  "algo" = "DSO", 
  "error" = ev_res[[49]]$pq_errors
)

pc_errors_DSM <- data.frame(
  "algo" = "DSM", 
  "error" = ev_res[[46]]$pq_errors
)

df_oi <- rbind(pc_errors_ORB, pc_errors_DSO, pc_errors_DSM)

g <- ggplot(data = df_oi, aes(error, color=algo)) + geom_density(size = 1) + xlim(c(0,0.7)) + xlab("Distance to closest point of ground truth in m")
ggsave(filename = "pc_distr.png", plot = g, path = "C:/Users/julia/Downloads", 
       width = 11, height = 6)

# trajectory

get_traj_df <- function(eval_list, seq_oi, alg_oi, run_oi){
  for (l in eval_list){
    if ((l$algorithm == alg_oi) && (l$sequence == seq_oi) && (l$run == run_oi)){
      return(data.frame(
        "algo" = alg_oi, 
        "run" = run_oi, 
        "seq" = seq_oi,
        "x" = l$traj_x,
        "y" = l$traj_y,
        "z" = l$traj_z
      ))
    }
  }
}

get_traj_df_gt <- function(eval_list, seq_oi, alg_oi, run_oi){
  for (l in eval_list){
    if ((l$algorithm == alg_oi) && (l$sequence == seq_oi) && (l$run == run_oi)){
      return(data.frame(
        "algo" = "ground_truth", 
        "run" = run_oi, 
        "seq" = seq_oi,
        "x" = l$traj_gt_x,
        "y" = l$traj_gt_y,
        "z" = l$traj_gt_z
      ))
    }
  }
}


get_traj_dist_df <- function(eval_list, seq_oi, alg_oi, run_oi){
  for (l in eval_list){
    if ((l$algorithm == alg_oi) && (l$sequence == seq_oi) && (l$run == run_oi)){
      return(data.frame(
        "algo" = alg_oi, 
        "run" = run_oi, 
        "seq" = seq_oi,
        "time" = l$traj_timestamp,
        "error" = l$traj_dist
      ))
    }
  }
}




get_seq_tray <- function(eval_list, seq_oi, alg_oi, run_oi){
  run1 <- get_traj_df(eval_list = eval_list, 
                      seq_oi = seq_oi, 
                      alg_oi = alg_oi,
                      run_oi = "data_0")
  run1$run=1
  run2 <- get_traj_df(eval_list = eval_list, 
                      seq_oi = seq_oi, 
                      alg_oi = alg_oi,
                      run_oi = "data_1")
  run2$run=2
  run3 <- get_traj_df(eval_list = eval_list, 
                      seq_oi = seq_oi, 
                      alg_oi = alg_oi,
                      run_oi = "data_2")
  run3$run=3
  return(
    rbind(run1, run2, run3)
  )
}

for (seq_oi in unique(sequences)){
  orb_mh01 <- get_traj_df(eval_list = ev_res, seq_oi = seq_oi, alg_oi = "ORB", run_oi = "data_1")
  dsm_mh01 <- get_traj_df(eval_list = ev_res, seq_oi = seq_oi, alg_oi = "DSM", run_oi = "data_1")
  dso_mh01 <- get_traj_df(eval_list = ev_res, seq_oi = seq_oi, alg_oi = "DSO", run_oi = "data_1")
  gt_df <- get_traj_df_gt(eval_list = ev_res, seq_oi = seq_oi, alg_oi = "DSO", run_oi = "data_1")
  
  
  test_df <- rbind(orb_mh01, dsm_mh01, dso_mh01, gt_df)
  
  
  g <- ggplot(test_df, aes(x=x, y=y, z=z, color=algo), size = c(1,1,1,3)) + 
    theme_void() +
    axes_3D() +
    stat_3D(geom="path") +
    labs_3D( hjust=c(1,0,0), vjust=c(1.5,1,-.2),
            labs=c("x", "y", "z")) 
  ggsave(filename = paste0("traj_", seq_oi, ".png"), plot = g, path = "C:/Users/julia/Downloads", 
         width = 11, height = 6) 
}



plot_tray_error_over_time <- function(ev_list, seq_oi, run_oi){
  # orb 
  orb <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "ORB", 
                          run_oi = run_oi)
  dsm <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSM", 
                          run_oi = run_oi)
  dso <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSO", 
                          run_oi = run_oi)
  
 df_oi <- rbind(orb, dsm, dso)
 g <- ggplot(df_oi) + geom_path(aes(x = time, y = error, color=algo))
 g
}

plot_tray_error_over_time(ev_list = ev_res, seq_oi = "MH01", run_oi = "data_0")


df_tray_error_over_time_comp <- function(ev_list, seq_oi){
  # orb 
  orb_1 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "ORB", 
                          run_oi = "data_0")
  orb_2 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "ORB", 
                            run_oi = "data_1")
  orb_3 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "ORB", 
                            run_oi = "data_2")
  # dsm
  dsm_1 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSM", 
                          run_oi = "data_0")
  dsm_2 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSM", 
                          run_oi = "data_1")
  dsm_3 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSM", 
                          run_oi = "data_2")

  # dso
  dso_1 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSO", 
                            run_oi = "data_0")
  dso_2 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSO", 
                            run_oi = "data_1")
  dso_3 <- get_traj_dist_df(eval_list = ev_list, seq_oi = seq_oi, alg_oi = "DSO", 
                            run_oi = "data_2")
  
  df_oi <- rbind(orb_1, orb_2, orb_3, dsm_1, dsm_2, dsm_3, dso_1, dso_2, dso_3)
  df_oi$run <- ifelse(test = df_oi$run == "data_0", yes = "1",
                      no =  ifelse(df_oi$run == "data_1", yes = "2", no = "3"))
  return(df_oi)
}

df_oi <- df_tray_error_over_time_comp(ev_list = ev_res, seq_oi = "V203")
g <- ggplot(data = df_oi, mapping = aes(x = time, y = error, color=run)) + geom_path() + facet_grid(rows = vars(algo))
g <- g + xlab("Time in s") + ylab("Euclidean distance to true position")
ggsave(filename = "comp_runs.png", plot = g, path = "C:/Users/julia/Downloads", 
       width = 11, height = 6)

df_oi <- df_tray_error_over_time_comp(ev_list = ev_res, seq_oi = "MH01")
g <- ggplot(data = df_oi, mapping = aes(x = time, y = error, color=run)) + geom_path() + facet_grid(rows = vars(algo))
g <- g + xlab("Time in s") + ylab("Euclidean distance to true position")
ggsave(filename = "comp_runs2.png", plot = g, path = "C:/Users/julia/Downloads", 
       width = 11, height = 6)
########################################
# Pointcloud
########################################

Seqs = unique(sequences)

point_errors_ORB <- sapply(Seqs, function(seq_oi){
  means <- c()
  for (run_oi in ev_res){
    if((run_oi$algorithm == "ORB")&& (run_oi$sequence==seq_oi)){
      means <- c(means, mean(run_oi$pq_errors))
    } 
  }
  print(means)
  return(round(mean(means), 4))
})

point_errors_DSM <- sapply(Seqs, function(seq_oi){
  means <- c()
  for (run_oi in ev_res){
    if((run_oi$algorithm == "DSM")&& (run_oi$sequence==seq_oi)){
      means <- c(means, mean(run_oi$pq_errors))
    } 
  }
  print(means)
  return(round(mean(means), 4))
})

point_errors_DSO <- sapply(Seqs, function(seq_oi){
  means <- c()
  for (run_oi in ev_res){
    if((run_oi$algorithm == "DSO")&& (run_oi$sequence==seq_oi)){
      means <- c(means, mean(run_oi$pq_errors))
    } 
  }
  print(means)
  return(round(mean(means), 4))
})

# number of points

points_ORB <- sapply(Seqs, function(seq_oi){
  means <- c()
  for (run_oi in ev_res){
    if((run_oi$algorithm == "ORB")&& (run_oi$sequence==seq_oi)){
      means <- c(means, run_oi$n_points)
    } 
  }
  print(means)
  return(round(mean(means), 4))
})

points_DSM <- sapply(Seqs, function(seq_oi){
  means <- c()
  for (run_oi in ev_res){
    if((run_oi$algorithm == "DSM")&& (run_oi$sequence==seq_oi)){
      means <- c(means, run_oi$n_points)
    } 
  }
  print(means)
  return(round(mean(means), 4))
})

points_DSO <- sapply(Seqs, function(seq_oi){
  means <- c()
  for (run_oi in ev_res){
    if((run_oi$algorithm == "DSO")&& (run_oi$sequence==seq_oi)){
      means <- c(means, run_oi$n_points)
    } 
  }
  print(means)
  return(round(mean(means), 4))
})


# computation time 

fps_df_agg <- fps_df %>% group_by(algorithm, seq) %>% summarize(mean(proc_time))








