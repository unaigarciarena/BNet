library(RcppCNPy)
library(bnlearn)
library(stringi)
library(comprehenr)
#setwd(getSrcDirectory()[1])

gdeletes = list(c(), c(FALSE, FALSE,  TRUE, FALSE, FALSE, FALSE), c(FALSE,  TRUE, FALSE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE, FALSE), c(FALSE, FALSE,  TRUE, FALSE,  TRUE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE,  TRUE, FALSE), c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,  TRUE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE, FALSE, FALSE, FALSE), c(FALSE,FALSE,FALSE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE))
ddeletes = list(c(), c(TRUE, FALSE,  TRUE, FALSE,  TRUE, FALSE), c(TRUE,  TRUE, FALSE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE, FALSE), c(FALSE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE,  TRUE, FALSE), c(FALSE,  TRUE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE,  TRUE,  TRUE, FALSE,  TRUE,  TRUE,  TRUE,  TRUE, FALSE))
deletes = list("generator"=gdeletes, "discriminator"=ddeletes)
variables = list(c(), c("L_Tp1","L_Tp2","Act1","Act2","Ch_Nn1","Ch_Nn2"), c("L_Tp1","L_Tp2","L_Tp3","Act1","Act2","Act3","Ch_Nn1","Ch_Nn2","Ch_Nn3"), c("L_Tp1","L_Tp2","L_Tp3","L_Tp4","Act1","Act2","Act3","Act4","Ch_Nn1","Ch_Nn2","Ch_Nn3","Ch_Nn4"), c("L_Tp1","L_Tp2","L_Tp3","L_Tp4","L_Tp5","Act1","Act2","Act3","Act4","Act5","Ch_Nn1","Ch_Nn2","Ch_Nn3","Ch_Nn4","Ch_Nn5"), c("L_Tp1","L_Tp2","L_Tp3","L_Tp4","L_Tp5","L_Tp6","Act1","Act2","Act3","Act4","Act5","Act6","Ch_Nn1","Ch_Nn2","Ch_Nn3","Ch_Nn4","Ch_Nn5","Ch_Nn6"))
depthmasks = list(c(TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE), c(TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE), c(TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE), c(TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE), c(TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE), c(TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE))

create.data.vars <-function(network, depth){
  
  path = paste(network, depth, "ForR.npy", sep="")
  
  variables = c(to_vec(for(x in 1:depth) paste("L_Tp", x, sep="")))
  discretizin = c(to_vec(for(x in 1:depth) 2))
  
  variables = c(variables, to_vec(for(x in 1:(depth)) paste("Act", x, sep="")))
  discretizin = c(discretizin, to_vec(for(x in 1:(depth)) 4))
  
  variables = c(variables, to_vec(for(x in 1:depth) paste("Ch_Nn", x, sep="")))
  discretizin = c(discretizin, to_vec(for(x in 1:depth) 5))
  
  nets = npyLoad(path, dotranspose = FALSE)
  nets = data.frame(nets)
  nets[,-(depth*2):-1] = log(nets[,-(depth*2):-1])
  delete = lapply(nets, var, na.rm = TRUE) != 0
  discretizin = discretizin[delete]
  reduced_variables = variables[delete]
  
  reduced_nets = nets[delete]
  reduced_nets = data.frame(reduced_nets)
  reduced_nets = discretize(reduced_nets, method="interval", breaks=discretizin)
  colnames(reduced_nets) = reduced_variables
  reduced_nets = data.frame(reduced_nets)
  
  return(list("Nets" = nets, "ReducedNets" = reduced_nets, "Reduced_Variables" = reduced_variables, "Variables" = variables, "Delete" = delete, "Discretizin" = discretizin))
}

load.neighbors <-function(neighbors, depths){
  
  path = paste(network, depth, "ForR.npy", sep="")
  
  variables = c(to_vec(for(x in 1:depth) paste("L_Tp", x, sep="")))
  discretizin = c(to_vec(for(x in 1:depth) 2))
  
  variables = c(variables, to_vec(for(x in 1:(depth)) paste("Act", x, sep="")))
  discretizin = c(discretizin, to_vec(for(x in 1:(depth)) 6))
  
  variables = c(variables, to_vec(for(x in 1:depth) paste("Ch_Nn", x, sep="")))
  discretizin = c(discretizin, to_vec(for(x in 1:depth) 5))
  
  nets = npyLoad(path, dotranspose = FALSE)
  nets = data.frame(nets)
  
  delete = lapply(nets, var, na.rm = TRUE) != 0
  discretizin = discretizin[delete]
  reduced_variables = variables[delete]
  
  reduced_nets = nets[delete]
  reduced_nets = data.frame(reduced_nets)
  reduced_nets = discretize(reduced_nets, method="interval", breaks=discretizin)
  colnames(reduced_nets) = reduced_variables
  reduced_nets = data.frame(reduced_nets)
  
  return(list("Nets" = nets, "ReducedNets" = reduced_nets, "Reduced_Variables" = reduced_variables, "Variables" = variables, "Delete" = delete, "Discretizing" = discretizin))
}

level.computer<-function(range, number){
  ones = ""
  if (length(dim(range))>2){
    for(i in 1:(length(dim(range))-2)){
      ones = paste(ones, ", 1", sep="")
    }
  }
  command = paste("names(range[1, ", ones, "])", sep="")
  nms = eval(parse(text = command))
  rngs = to_vec(for(x in nms) stri_sub(x, 2, -2))
  index = 1
  for (rng in rngs){
    a = strsplit(rng, split="_")[[1]]
    if (number>=as.numeric(a[1]))
      if (number<=as.numeric(a[2]))
        break
    index = index+1
  }
  if(index==1)
    return(paste("'[", rngs[index], "]'", sep=""))
  else
    return(paste("'_", rngs[index], "]'", sep=""))
}

single.level.computer<-function(range, number){
  nms = names(range)
  rngs = to_vec(for(x in nms) stri_sub(x, 2, -2))
  index = 1
  for (rng in rngs){
    a = strsplit(rng, split="_")[[1]]
    if (number>=as.numeric(a[1]))
      if (number<=as.numeric(a[2]))
        break
    index = index+1
  }
  if(index==1)
    return(paste("[", rngs[index], "]", sep=""))
  else
    return(paste("_", rngs[index], "]", sep=""))
}

learn.bn <-function(data, variables, net, depth){
  pdag = chow.liu(x=data, mi="mi")
  dag = pdag2dag(pdag, ordering = variables)
  plot(dag)
  fit = bn.fit(dag, data, method = "bayes")
  write.net(paste(net, depth, ".hugin", sep=""), fit)
  return(fit)
}

create.nets <- function(){
  for(net in c("generator", "discriminator")){
    for(depth in 2:6){
      gens = create.data.vars(net, depth)
      fit = learn.bn(gens$ReducedNets, gens$Reduced_Variables, net, depth)
    }
  }
}

load.nets <- function(net){
  fits = list()
  for(depth in 2:6){
    if(net!="discriminator"|depth!=6)
      fits[[depth]] = read.net(paste(net, depth, ".hugin", sep=""))
  }
  return(fits)
}

manual.probas <- function(res, fits){
  data = res$Inds
  depths = res$Depths
  command = ""
  probas = c()
  for (obs in 1:length(data)){
    event = data[[obs]]
    fit = fits[[depths[obs]]]
    unsampled = colnames(event)
    prob = 1
    while(length(unsampled)>0){
      for (var in unsampled){
        independent = TRUE
        aux = fit[[var]]$prob
        for (parent in fit[[var]]$parents){
          
          if (parent %in% unsampled){
            independent = FALSE
            break
          }
          else{
            commas = ""
            for(i in 1:(length(dim(aux))-1))
              commas = paste(commas, ",", sep="")
            stri_sub(commas, -1, -1) = ""
            temp = level.computer(aux, event[1,parent])
            
            command = paste("aux[,", temp, commas, "]", sep="")
            
            if(grepl("NA", command, fixed=TRUE))
              break
            aux = eval(parse(text = command))
          }
          
        }
        if(grepl("NA", command, fixed=TRUE))
          break
        if (independent){

          temp = single.level.computer(aux, event[1,var])
          if(grepl("NA", temp, fixed=TRUE))
            break
          prob = prob*aux[temp]
          
          unsampled = unsampled[unsampled != var]
        }
      }
      if(grepl("NA", command, fixed=TRUE)|grepl("NA", temp, fixed=TRUE)){

        command = ""
        temp = ""
        prob = runif(1, -2, -1)
        break
      }
    }
    probas = c(prob[[1]], probas)
    prob = 1
  }
  return(probas)
}

compute.lengths<-function(path, type){
  inds = c()
  nets = npyLoad(path, dotranspose = TRUE)
  depths = rowSums((nets[,1:6]>-1)*1)
  for(inet in 1:nrow(nets)){
    depth = depths[[inet]]

    if(type=="discriminator"&depth==6)
      break
    dpthmsk = depthmasks[[depth]]
    net = nets[inet,dpthmsk]
    delete = deletes[[type]][[depth]]
    net = net[delete]
    net = data.frame(rbind(net))
    net[,-(dim(net)[2]-depth+1):-1] = log(net[,-(dim(net)[2]-depth+1):-1])
    colnames(net) = variables[[depth]][delete]
    # nets[,-(depth*2):-1] = log(nets[,-(depth*2):-1])
    inds = c(inds, list(net))
  }
  return(list("Inds"=inds, "Depths"=depths))
}