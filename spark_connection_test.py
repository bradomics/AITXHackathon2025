import traci
traci.init(8813, host="localhost")
print("time:", traci.simulation.getTime())
print("loaded:", traci.simulation.getLoadedNumber())
print("departed:", traci.simulation.getDepartedNumber())
print("arrived:", traci.simulation.getArrivedNumber())
print("minExpected:", traci.simulation.getMinExpectedNumber())
print("vehIDs:", traci.vehicle.getIDList()[:20])
traci.close()
