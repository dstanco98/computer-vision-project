# computer-vision-project
Project of Computer Vision course, A.Y. 2022/2023, Sapienza University of Rome.

## Offside detection system for football

As we know, the World Cup took place in Qatar in December 2022, during which a new technology was introduced, that of <b>semi-automated offside</b>, where the <b>key points</b> of each players are detected and the "offside line" is drawn at the last player of the defending team. The following image represent the system introduced by the FIFA.
  ![example of FIFA semi-automated offside](images/offside.jpeg)
The main purpose of this project is to "replicate" in some way this behaviour but since it requires the combination of some powerful technologies, indeed the offside detection problem is <b>simplified by classifying the attacking and defending team based on the half in which the foward ball is played</b>. Further the assumption that team members wear the same coloured jerseys works towards simplifying the problem.
The implementation involves two separate modules that track the ball and players respectively. The successful integration of the modules leads to the desired goal of offside detection.
  

## Approach
* The problem was broken down into a **ball tracking module** and a  **player tracking module** then combining the two to detect Offside. 
* The **ball tracking module** would take care of detecting the ball, tracking it and detecting whether a ball pass has occurred. 
* The **player tracking module** would detect the players of each team, attacking and defending, and get an approximate location of the foot of the players. 
* Finally the two would be integrated into one program. The ball pass is detected only when it is passed from one player to different player. 
* If the player of the attacking team receiving the pass (when he receives the pass) is behind the last player of the defending team then offside is called.
* The offside region is shown by a line passing through the position of the last defender.
* Note that offside is NOT called if the attacking player is behind the offside line but doesn’t receive the ball.


## Usage of the system:
* To detect offside in a pre-recorded video, go to the folder in which code and video are stored and type the following in terminal:
```javascript
$ python offside.py -v 'path/name of video file'
```
* To detect offside from live camera feed:
```javascript
$ python offside.py
```
While running the program, press 'i' to input and 'q' to quit:
* input patch of jersey of any player of team A
* input patch of jersey of any player of team B
* input patch of the ball
* input two points along each side of the field in this order, i.e. Top edge, Left edge, Bottom edge and Right edge

## Slides
It is possible to find the slides of the presentation of the project at the following link [Computer Vision Slides](https://docs.google.com/presentation/d/1ds1fc1_Nq5ZkOHw5Zt2-u0oVsnVPieCfXAnOmABnuyQ/edit?usp=sharing).

## References
Reference Paper:
* [Vision-Based-Dynamic-Offside-Line-Marker-for-Soccer-Games](https://arxiv.org/abs/1804.06438)

References for the implementation:
* [PassNet](https://github.com/jonpappalord/PassNet)
* [ITPS-Project](https://github.com/kparth98/ITSP-Project)
* [Computer-Vision-based-Offside-Detection-in-Soccer](https://github.com/Neerajj9/Computer-Vision-based-Offside-Detection-in-Soccer)
