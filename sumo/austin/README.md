NOTE: you must run:
- SUMO first (with sumo-gui -c sim.sumocfg --start --remote-port 8813)
- Then the web socket server
- Then make a call to the front end at localhost:3000/dashboard

to generate new random trips: python3 "$SUMO_HOME/tools/randomTrips.py" \                      
  -n austin.net.xml \     
  -r routes.rou.xml \
  -e 3600 \
  --seed 42 \
  --prefix veh \
  --fringe-factor 10 \
  --min-distance 300

to generate new austin.net.xml:
netconvert \                                      
  --osm-files austin.osm \
  --output-file austin.net.xml \
  --osm.all-attributes true \
  --roundabouts.guess true \
  --ramps.guess true \
  --junctions.join true \
  --proj.simple true
  --proj.utm

to run SUMO simulation in parallel with traCI and web socket server: sumo-gui -c sim.sumocfg --start --remote-port 8813
