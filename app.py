from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import plotly.graph_objects as go
import datetime
from dataclasses import dataclass

# Type aliases for clarity
TimeType = datetime.time
DateTimeType = datetime.datetime

# Constants
SIMULATION_START: DateTimeType = datetime.datetime.now().replace(
    hour=6, minute=0, second=0, microsecond=0
)
SIMULATION_END: DateTimeType = SIMULATION_START + datetime.timedelta(days=1)
INTERVALS: pd.DatetimeIndex = pd.date_range(SIMULATION_START, periods=49, freq="30min")
CHARGING_WINDOW_START: DateTimeType = SIMULATION_START.replace(hour=23, minute=30)
CHARGING_WINDOW_END: DateTimeType = SIMULATION_END.replace(hour=5, minute=30)

# Visualization constants
RED: str = "#ff3333"
LIGHT_BLUE: str = "#22d3ee"

# Charging constants
CHARGING_POWER: float = 3.5  # kW (per 30-minute interval)
MIN_PLUG_OUT_HOUR: int = 6  # 6 AM


# Default configurations for each archetype
ARCHETYPE_CONFIGS = {
    "Average UK": {
        "name": "Average UK",
        "battery_kwh": 60,
        "plug_in_frequency": 1.0,
        "plug_in_hour": 18,
        "plug_in_minute": 0,
        "plug_out_hour": 7,
        "plug_out_minute": 0,
        "target_soc": 0.8,
        "plug_in_soc": 0.68,
    },
    "Intelligent Octopus": {
        "name": "Intelligent Octopus",
        "battery_kwh": 72.5,
        "plug_in_frequency": 1.0,
        "plug_in_hour": 18,
        "plug_in_minute": 0,
        "plug_out_hour": 7,
        "plug_out_minute": 0,
        "target_soc": 0.8,
        "plug_in_soc": 0.52,
    },
    "Infrequent Charging": {
        "name": "Infrequent Charging",
        "battery_kwh": 60,
        "plug_in_frequency": 0.2,
        "plug_in_hour": 18,
        "plug_in_minute": 0,
        "plug_out_hour": 7,
        "plug_out_minute": 0,
        "target_soc": 0.8,
        "plug_in_soc": 0.18,
    },
    "Infrequent Driving": {
        "name": "Infrequent Driving",
        "battery_kwh": 60,
        "plug_in_frequency": 1.0,
        "plug_in_hour": 18,
        "plug_in_minute": 0,
        "plug_out_hour": 7,
        "plug_out_minute": 0,
        "target_soc": 0.8,
        "plug_in_soc": 0.73,
    },
    "Scheduled Charging": {
        "name": "Scheduled Charging",
        "battery_kwh": 60,
        "plug_in_frequency": 1.0,
        "plug_in_hour": 22,
        "plug_in_minute": 0,
        "plug_out_hour": 9,
        "plug_out_minute": 0,
        "target_soc": 0.8,
        "plug_in_soc": 0.68,
    },
    "Always Plugged In": {
        "name": "Always Plugged In",
        "battery_kwh": 60,
        "plug_in_frequency": 1.0,
        "plug_in_hour": 0,
        "plug_in_minute": 0,
        "plug_out_hour": 23,
        "plug_out_minute": 59,
        "target_soc": 0.8,
        "plug_in_soc": 0.68,
    },
}


@dataclass
class GlobalVariance:
    """Global configuration for behavioral variance parameters.

    Attributes:
        plug_in_time_std: Standard deviation for plug-in time in hours
        plug_out_time_std: Standard deviation for plug-out time in hours
        soc_std_factor: Standard deviation for SoC as a fraction of base SoC
    """

    plug_in_time_std: float = 0.0
    plug_out_time_std: float = 0.0
    soc_std_factor: float = 0.00


class EVUser:
    """
    Represents an electric vehicle user with specific charging behaviors.

    Attributes:
        soc_history (List[float]): History of state of charge values
        battery_kwh (float): Battery capacity in kWh
        plug_in_frequency (float): Probability of plugging in on any given day
        target_soc (float): Target state of charge
        plug_in_soc (float): State of charge when plugging in
        plug_in_time (TimeType): Time when vehicle is typically plugged in
        plug_out_time (TimeType): Time when vehicle is typically unplugged
    """

    def __init__(self, config: dict):
        """
        Initialize an EV user with given configuration.

        Args:
            config (dict): Configuration dictionary containing charging behavior parameters
        """
        self.name = config["name"]
        self.soc_history: List[float] = []
        self.battery_kwh = config["battery_kwh"]
        self.plug_in_frequency = config["plug_in_frequency"]
        self.target_soc = config["target_soc"]
        self.plugs_in = True

        self.plug_in_soc = self._calculate_plug_in_soc(config["plug_in_soc"])

        self.plug_in_time = self._calculate_plug_in_time(
            config["plug_in_hour"], config["plug_in_minute"]
        )

        self.plug_out_time = self._calculate_plug_out_time(
            config["plug_out_hour"], config["plug_out_minute"]
        )

    def _calculate_plug_in_soc(self, base_soc: float) -> float:
        """
        Calculate plug-in SoC with random variation.

        Args:
            base_soc (float): Base state of charge value

        Returns:
            float: Calculated plug-in SoC with random variation
        """
        if self.name == "Always Plugged In":
            return base_soc

        std_dev = st.session_state.variance.soc_std_factor * base_soc
        plug_in_soc = random.gauss(base_soc, std_dev)
        return max(0.05, min(0.95, plug_in_soc))

    def _calculate_plug_in_time(self, base_hour: int, base_minute: int) -> TimeType:
        """
        Calculate plug-in time with random variation. The minimum plug-in time is 12 PM, and the maximum is 11:30 PM.

        Args:
            base_hour (int): Base hour for plug-in time
            base_minute (int): Base minute for plug-in time

        Returns:
            TimeType: Calculated plug-in time with random variation
        """
        if self.name == "Always Plugged In":
            return datetime.time(base_hour, base_minute)

        variation = random.gauss(
            0, st.session_state.variance.plug_in_time_std
        ) 
        adjusted_hour = max(12, min(23, round(base_hour + variation)))
        return datetime.time(adjusted_hour, base_minute)

    def _calculate_plug_out_time(self, base_hour: int, base_minute: int) -> TimeType:
        """
        Calculate plug-out time with random variation. The minimum plug-out time is 6 AM, and the maximum is 12 PM.

        Args:
            base_hour (int): Base hour for plug-out time
            base_minute (int): Base minute for plug-out time

        Returns:
            TimeType: Calculated plug-out time with random variation
        """
        if self.name == "Always Plugged In":
            return datetime.time(base_hour, base_minute)

        variation = random.gauss(
            0, st.session_state.variance.plug_out_time_std
        )
        adjusted_hour = max(MIN_PLUG_OUT_HOUR, min(12, round(base_hour + variation)))
        return datetime.time(adjusted_hour, base_minute)

    def _calculate_charging_periods(self, charge_needed: float) -> List[float]:
        """
        Calculate charging periods based on needed charge.

        Args:
            charge_needed (float): Amount of charge needed in kWh

        Returns:
            List[float]: List of charging amounts for each period
        """
        num_periods_needed = min(math.ceil(charge_needed / CHARGING_POWER), 12)
        charging_window = int((CHARGING_WINDOW_END - CHARGING_WINDOW_START).total_seconds() / 1800)  # Number of 30-minute intervals in the charging window

        charging_periods = [CHARGING_POWER] * num_periods_needed + [0] * (
            charging_window - num_periods_needed
        )
        random.shuffle(charging_periods)
        return charging_periods

    def simulate_day(self) -> None:
        """
        Simulate a day of battery SoC, starting at 6 AM and ending at 6 AM the next day.
        Updates the soc_history attribute with simulated values.
        """
        # Simulate until plug-out time (do not charge)
        num_periods = (self.plug_out_time.hour - MIN_PLUG_OUT_HOUR) * 2
        self.soc_history.extend(self.target_soc for _ in range(num_periods + 1))

        # Simulate discharge until plug-in time
        self._simulate_discharge()

        # Simulate until charging window (do not charge or discharge)
        charging_window_start = CHARGING_WINDOW_START.hour + CHARGING_WINDOW_START.minute / 60
        num_periods = int((charging_window_start - self.plug_in_time.hour) * 2)
        self.soc_history.extend(self.soc_history[-1] for _ in range(num_periods))

        # Simulate charging or non-charging behavior
        self._simulate_charging()

    def _simulate_discharge(self) -> None:
        """Simulate discharge period between plug-out and plug-in times."""
        num_periods = (self.plug_in_time.hour - self.plug_out_time.hour) * 2
        step_size = (self.plug_in_soc - self.target_soc) / int(num_periods)

        for _ in range(num_periods):
            self.soc_history.append(round(self.soc_history[-1] + step_size, 4))

    def _simulate_charging(self) -> None:
        """Simulate charging behavior during the charging window."""
        if random.random() > self.plug_in_frequency:
            # Not charging tonight
            self.soc_history.extend([self.soc_history[-1]] * 13)  # 13 periods in charging window
            self.plugs_in = False
            return

        charge_needed = self.battery_kwh * (self.target_soc - self.plug_in_soc)
        charging_periods = self._calculate_charging_periods(charge_needed)

        charge_list = (
            np.cumsum(charging_periods) / self.battery_kwh
        ) + self.soc_history[-1]
        charge_list = np.minimum(charge_list, self.target_soc)

        self.soc_history.extend(np.round(charge_list, 4))
        self.soc_history.append(self.soc_history[-1])  # Add final period until 6 AM (do not charge)


def config_section(archetype: str, config: dict) -> dict:
    """
    Create configuration section for an archetype with input validation.

    Args:
        archetype (str): Name of the archetype
        config (dict): Current configuration for the archetype

    Returns:
        dict: Updated configuration based on user input
    """
    new_config = config.copy()

    col1, col2 = st.columns(2)
    with col1:
        new_config["battery_kwh"] = st.slider(
            "Battery Capacity (kWh)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=float(config["battery_kwh"]),
            key=f"{archetype}_battery",
            format="%f",
            help="Battery capacity in kWh",
        )

        new_config["plug_out_hour"] = st.slider(
            "Plug-out Hour",
            min_value=6,
            max_value=12,
            step=1,
            value=int(config["plug_out_hour"]),
            key=f"{archetype}_out_hour",
            help="Time when vehicle is typically unplugged",
        )

        new_config["target_soc"] = st.slider(
            "Target SoC",
            min_value=0.0,
            max_value=1.0,
            value=float(config["target_soc"]),
            key=f"{archetype}_target_soc",
            help="Target state of charge",
        )

    with col2:
        new_config["plug_in_frequency"] = st.slider(
            "Plug-in Frequency",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=float(config["plug_in_frequency"]),
            key=f"{archetype}_frequency",
            help="Probability of plugging in on any given day",
        )

        new_config["plug_in_hour"] = st.slider(
            "Plug-in Hour",
            min_value=12,
            max_value=23,
            step=1,
            value=int(config["plug_in_hour"]),
            key=f"{archetype}_in_hour",
            help="Time when vehicle is typically plugged in",
        )

        new_config["plug_in_soc"] = st.slider(
            "Plug-in SoC",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(config["plug_in_soc"]),
            key=f"{archetype}_plug_soc",
            help="State of charge when vehicle is plugged in",
        )

    return new_config


def archetype_overview(user: EVUser) -> None:
    """
    Display key metrics for an EV user archetype.

    Args:
        user (EVUser): EVUser instance to display metrics for
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Battery Capacity", f"{user.battery_kwh} kWh")
        st.metric("Plug-in Frequency", f"{round(user.plug_in_frequency * 100)}%")
    with col2:
        st.metric("Plug-out Time", user.plug_out_time.strftime("%H:%M"))
        st.metric("Target SoC", f"{user.target_soc:.0%}")
    with col3:
        if user.plugs_in:
            st.metric("Plug-in Time", user.plug_in_time.strftime("%H:%M"))
        else:
            st.metric("Plug-in Time", "N/A")
        st.metric("Plug-in SoC", f"{user.plug_in_soc:.0%}")


def plot_agent(user: EVUser, archetype: str) -> go.Figure:
    """
    Create a plot showing the state of charge for a single EV agent.

    Args:
        user (EVUser): EVUser instance to plot
        archetype (str): Name of the archetype being plotted

    Returns:
        go.Figure: Plotly figure object containing the plot
    """
    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=SIMULATION_START.replace(hour=23, minute=30),
        x1=SIMULATION_END.replace(hour=5, minute=30),
        y0=0,
        y1=1,
        fillcolor="lightgreen",
        opacity=0.2,
        layer="below",
        line_width=0
    )

    fig.add_annotation(
        x=SIMULATION_END.replace(hour=2, minute=30),
        y=0.975,
        text="Charging Window",
        showarrow=False,
    )

    fig.add_trace(
        go.Scatter(
            x=INTERVALS,
            y=user.soc_history,
            mode="lines+markers",
            line=dict(color=LIGHT_BLUE),
        )
    )

    # Do not show plug-in times if the archetype is always plugged in
    if archetype != "Always Plugged In":
        fig.add_vline(
            x=SIMULATION_START.replace(
                hour=user.plug_out_time.hour, minute=user.plug_out_time.minute
            ).timestamp()
            * 1000,  # Convert to milliseconds for plotly
            line_width=2,
            line_dash="dash",
            line_color=RED,
            annotation_text="Plug-out Time",
            annotation_position="top",
        )

        if user.plugs_in:
            fig.add_vline(
                x=SIMULATION_START.replace(hour=user.plug_in_time.hour).timestamp()
                * 1000,  # Convert to milliseconds for plotly
                line_width=2,
                line_dash="dash",
                line_color=RED,
                annotation_text="Plug-in Time",
                annotation_position="top",
            )

    fig.update_layout(
        title="State of Charge for One Agent",
        xaxis_title="Time",
        yaxis_title="State of Charge (%)",
        xaxis_tickformat="%H:%M",
        yaxis=dict(tickformat=",.0%"),
        xaxis_range=[SIMULATION_START, SIMULATION_END],
        yaxis_range=[0, 1],
    )

    return fig


def plot_population(population: List[EVUser]) -> go.Figure:
    """
    Create a plot showing aggregate metrics for a population of EV users.

    Args:
        population (List[EVUser]): List of EVUser instances to analyze

    Returns:
        go.Figure: Plotly figure object containing the population plot
    """
    population_size = len(population)

    plugged_in = [
        sum(
            1
            for ev in population
            if (ev.plug_in_time <= t.time() or t.time() <= ev.plug_out_time) and ev.plugs_in
        )
        for t in INTERVALS
    ]
    proportion_plugged_in = [x / population_size for x in plugged_in]
    soc_df = pd.DataFrame(
        [ev.soc_history for ev in population], index=range(len(population))
    )

    # Plot population metrics
    fig = go.Figure()

    # Add bar chart for proportion plugged in
    fig.add_trace(
        go.Bar(
            x=INTERVALS,
            y=proportion_plugged_in,
            marker_color="lightgreen",
            opacity=0.5,
            name="Proportion Plugged In",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=INTERVALS,
            y=soc_df.mean(),
            mode="lines",
            line=dict(color=LIGHT_BLUE),
            name="Mean SoC",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=INTERVALS,
            y=soc_df.quantile(0.95),
            mode="lines",
            line=dict(color=LIGHT_BLUE, dash="dash"),
            name="95th Percentile",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=INTERVALS,
            y=soc_df.quantile(0.05),
            mode="lines",
            line=dict(color=LIGHT_BLUE, dash="dash"),
            name="5th Percentile",
        )
    )

    fig.add_vline(
        x=SIMULATION_START.replace(hour=23, minute=30).timestamp() * 1000,
        line_width=2,
        line_dash="dash",
        line_color=RED,
        annotation_text="Start Charging",
        annotation_position="top",
    )

    fig.add_vline(
        x=SIMULATION_END.replace(hour=5, minute=30).timestamp() * 1000,
        line_width=2,
        line_dash="dash",
        line_color=RED,
        annotation_text="End Charging",
        annotation_position="top",
        annotation_xshift=-20,
    )

    fig.update_layout(
        title="Population Metrics",
        xaxis_title="Time",
        yaxis_title="Proportion / State of Charge",
        xaxis_tickformat="%H:%M",
        yaxis=dict(tickformat=",.0%"),
        xaxis_range=[SIMULATION_START, SIMULATION_END],
        yaxis_range=[0, 1],
        legend=dict(
            yanchor="top",
            y=1.26,
            xanchor="left",
            x=0.15,
        )
    )

    return fig


def main():
    st.title("EV Charging Behavior Simulator")

    # Store the configurations in session state if not already present
    if "configs" not in st.session_state:
        st.session_state.configs = ARCHETYPE_CONFIGS.copy()

    if "variance" not in st.session_state:
        st.session_state.variance = GlobalVariance()

    st.sidebar.header("Simulation Parameters")

    with st.sidebar.expander("Global Variance Controls"):
        plug_in_time_std = st.slider(
            "Plug-in Time Variance (hours)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.variance.plug_in_time_std,
            step=0.1,
            help="Standard deviation of plug-in time variation in hours",
        )
        plug_out_time_std = st.slider(
            "Plug-out Time Variance (hours)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.variance.plug_out_time_std,
            step=0.1,
            help="Standard deviation of plug-out time variation in hours",
        )
        soc_std_factor = st.slider(
            "SoC Variance Factor",
            min_value=0.0,
            max_value=0.2,
            value=st.session_state.variance.soc_std_factor,
            step=0.01,
            help="Standard deviation of SoC variation as a fraction of base SoC",
        )

        st.session_state.variance.plug_in_time_std = plug_in_time_std
        st.session_state.variance.plug_out_time_std = plug_out_time_std
        st.session_state.variance.soc_std_factor = soc_std_factor

    with st.sidebar.expander("Archetype Configurations"):
        archetype = st.selectbox(
            "Select archetype to configure", list(ARCHETYPE_CONFIGS.keys())
        )

        st.session_state.configs[archetype] = config_section(
            archetype, st.session_state.configs[archetype]
        )

    with st.sidebar.expander("Population Distribution"):
        population_size = st.slider(
            "Population Size",
            min_value=1000,
            max_value=5000,
            value=2500,  # Same as CNZ report
            step=100,
            help="Total number of EVs in the simulation",
        )

        col1, col2 = st.columns(2)

        with col1:
            average_uk = st.slider("Average UK %", min_value=0, max_value=100, value=40)
            infrequent_charging = st.slider(
                "Infrequent Charging %", min_value=0, max_value=100, value=10,
            )
            scheduled_charging = st.slider(
                "Scheduled Charging %", min_value=0, max_value=100, value=9
            )

        with col2:
            intelligent_octopus = st.slider(
                "Intelligent Octopus %", min_value=0, max_value=100, value=30
            )
            infrequent_driving = st.slider(
                "Infrequent Driving %", min_value=0, max_value=100, value=10
            )
            always_plugged_in = st.slider(
                "Always Plugged In %", min_value=0, max_value=100, value=1
            )

    if (
        average_uk
        + intelligent_octopus
        + infrequent_charging
        + infrequent_driving
        + scheduled_charging
        + always_plugged_in
        != 100
    ):
        st.error("The sum of the percentages must equal 100.")
        st.stop()

    st.header("Individual Archetype Overview")

    archetype = st.selectbox("Select archetype to view", list(ARCHETYPE_CONFIGS.keys()))

    # Create and simulate selected archetype
    user = EVUser(st.session_state.configs[archetype])
    user.simulate_day()

    archetype_overview(user)
    st.plotly_chart(plot_agent(user, archetype))

    st.header("Agent Simulation")

    population_map = {
        "Average UK": average_uk,
        "Intelligent Octopus": intelligent_octopus,
        "Infrequent Charging": infrequent_charging,
        "Infrequent Driving": infrequent_driving,
        "Scheduled Charging": scheduled_charging,
        "Always Plugged In": always_plugged_in,
    }

    population = []
    for archetype, percentage in population_map.items():
        count = int(population_size * percentage / 100)
        population.extend(
            [EVUser(st.session_state.configs[archetype]) for _ in range(count)]
        )

    for agent in population:
        agent.simulate_day()

    st.plotly_chart(plot_population(population))

    st.header("Notes")

    st.markdown(
        """
    ### Assumptions
    - The battery is charged and discharged at a constant rate (which is inconsistent with real-world charging behavior).
    - The battery is charged in 30-minute intervals (based on CNZ report).
    - Charging only occurs between 11:30 PM and 5:30 AM (same as Intelligent Octopus).
    - Plug-out time is between 6 AM and 12 PM.
    - Plug-in time is between 12 PM and 11:30 PM.
    - Charging occurs uniformly randomly distributed throughout the charging window.
    - Driving is simulated as a linear interpolation between plug-out time and SoC and plug-in time and SoC.
    """
    )

    st.markdown(
        """
    ### Notes
    - The focus is to simulate a variety of scenarios, where many of the configurations can be adjusted.
    - The code was written to be modular and easily extensible, for example, adding new archetypes.
    - The simulation is based on a single day, not differentiating between weekdays and weekends.
    - Datetime calculations are not handled gracefully, and make many assumptions.
    - Little to no validation is performed on the input values.
    - Validation of results was done manually, additional graphs and metrics could be added for further validation.
    - Charging is limited to 42 kWh (6 hours * 7 kW) per night.
    """
    )

    # TODO: Mark users as not plugged in if they are not charging according to the plug-in frequency

if __name__ == "__main__":
    main()
