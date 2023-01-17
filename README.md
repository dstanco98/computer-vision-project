# computer-vision-project
Project of Computer Vision course, A.Y. 2022/2023, Sapienza University of Rome.

## Semi-automatic offside detection for Football

## Abstract

Here, I present my attempt toward the creation of a mechanism by which the problem of incorrect offside decisions made by referees in football can be addressed. The purpose of this system is to replicate the operation of the semi-automatic offside introduced in the last edition of the <b>Fifa World Cup in Qatar</b> but with some assumptions and restrictions. The offside detection problem is simplified by classifying the attacking and defending team based on the half in which the foward ball is played. Further the assumption that team members wear the same coloured jerseys works towards simplifying the problem.
The implementation involves two separate modules that track the ball and players respectively. The successful integration of the modules leads to the desired goal of offside detection.

## Approach
* The problem was broken down into a **ball tracking module** and a  **player tracking module** then combining the two to detect Offside. 
* The **ball tracking module** would take care of detecting the ball, tracking it and detecting whether a ball pass has occurred. 
* The **player tracking module** would detect the players of each team, attacking and defending, and get an approximate location of the foot of the players. 
* Finally the two would be integrated into one program. The ball pass is detected only when it is passed from one player to different player. 
* If the player of the attacking team receiving the pass (when he receives the pass) is behind the last player of the defending team then offside is called.
* The offside region is shown by a line passing through the position of the last defender.
* Note that offside is NOT called if the attacking player is behind the offside line but doesn’t receive the ball.

## Usage of the sysyem:
