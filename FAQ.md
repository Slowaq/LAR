---
title: Turtlebot FAQ
author: Libor Wagner <libor.wagner@cvut.cz>
date: 2020-02-20
---


## Singularity: unable to kill instance

 - **problem**: I have called `singularity instance stop [name]` but is still appears in the `singularity instance list`?
 - **solution**: remove the following directory `~/.singularity/instances/sign/[hostname]/[username]/[instancename]`
