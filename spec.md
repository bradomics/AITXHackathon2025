# Traffic Incident Insights

**The Problem:** Right now, Austin’s emergency response is completely reactive. A tow truck sits at a depot until a 911 call comes in, and by the time they get to I-35, the traffic is already backed up for miles. We can assume that crashes aren't random; they follow patterns based on the time of day, rain, and heat. It would be incredibly valuable if we could "see the future" of the grid and position safety assets near high-risk corridors *before* the incidents actually happen.

**The Goal:** Build a system that helps derive actionable insights from traffic incident reports (using weather data is optional).

### Dataset

[**Austin Real-Time Traffic Incident Reports](https://data.austintexas.gov/Transportation-and-Mobility/Real-Time-Traffic-Incident-Reports/dx9v-zd7x) (Live & Historical: Crashes, Stalls, Hazards)**

*Optional Enrichment:* [NOAA Weather API](https://www.weather.gov/documentation/services-web-api)

<aside>
<img src="/icons/light-bulb_orange.svg" alt="/icons/light-bulb_orange.svg" width="40px" />

### About

This dataset contains traffic incident information from the Austin-Travis County traffic reports collected from the various Public Safety agencies through a data feed from the Combined Transportation, Emergency, and Communications Center (CTECC).

For further context, see:

Active Incidents: Map and Context -

https://data.austintexas.gov/stories/s/Austin-Travis-County-Traffic-Report-Page/9qfg-4swh/

Data Trends and Analysis -

https://data.austintexas.gov/stories/s/48n7-m3me

The dataset is updated every 5 minutes with the latest snapshot of active traffic incidents.

</aside>

**Suggested Directions (Inspiration Only):**

- ***Predictive Alerting:* Forecast high-risk "Hotspots" for the next hour to stage tow trucks early.**
- ***Contextual Intelligence:* Correlate weather events (Rain/Ice) with crash types to change deployment strategies.**
- ***Digital Twin:* Simulate how traffic would flow if you proactively closed a dangerous ramp.**

**Examples:**

- **Hotspot Agent Watcher:** Build a system that learns the "Rhythm of the City" to predict hotspots based on time/day and visualizes where units should be stationed. An agent that divides Austin into a grid and assigns a "Risk Score" to each sector for every hour of the day.
- **Weather Watcher:** Build a system that understands *context*. It correlates traffic incidents with historical weather data to predict how rain, heat, or freezing conditions radically change safety risks. An agent that ingests the *current* weather forecast and modifies the standard deployment plan.

<aside>
<img src="/icons/code_orange.svg" alt="/icons/code_orange.svg" width="40px" />

### API Access & Usage

1. Make account here and create App Token in [Developer Settings](https://evergreen.data.socrata.com/profile/edit/developer_settings): https://evergreen.data.socrata.com/ 
2. [**API Documentation Here**](https://dev.socrata.com/foundry/data.austintexas.gov/dx9v-zd7x)
3. [**Read about using queries with SODA 3**](https://dev.socrata.com/docs/queries/)
</aside>