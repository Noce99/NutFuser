# This file define the get_a_random_weather function that is used to get a random weather for a simulation
import random
try:
    import carla
except:
    pass


def get_a_random_weather():
    cloudiness = random.betavariate(alpha=1, beta=4) * 100
    precipitation = random.betavariate(alpha=1.5, beta=5) * 100
    wind_intensity = random.betavariate(alpha=2, beta=2) * 100
    sun_altitude_angle = random.uniform(-90, 90)
    fog_density = random.betavariate(alpha=1, beta=4) * 100

    a_weather = carla.WeatherParameters(
        cloudiness=cloudiness,                  # 0 is a clear sky, 100 complete overcast (default: 0.0)
        precipitation=precipitation,            # 0 is no rain at all, 100 a heavy rain (default: 0.0)
        precipitation_deposits=precipitation,   # 0 means no puddles on the road, 100 means roads completely capped by
                                                # rain (default: 0.0)
        wind_intensity=wind_intensity,          # 0 is calm, 100 a strong wind (default: 0.0)
        sun_azimuth_angle=0.0,                  # 0 is an arbitrary North, 180 its corresponding South (default: 0.0)
        sun_altitude_angle=sun_altitude_angle,  # 90 is midday, -90 is midnight (default: 0.0)
        fog_density=fog_density,                # Concentration or thickness of the fog, from 0 to 100 (default: 0.0)
        fog_distance=15.0,                      # Distance where the fog starts in meters (default: 0.0)
        wetness=0.0,                            # Humidity percentages of the road, from 0 to 100 (default: 0.0)
        fog_falloff=0.0,                        # Density (specific mass) of the fog, from 0 to infinity (default: 0.0)
        scattering_intensity=0.0,               # Controls how much the light will contribute to volumetric fog.
                                                # When set to 0, there is no contribution (default: 0.0)
        mie_scattering_scale=0.0,               # Controls interaction of light with large particles like pollen or air
                                                # pollution resulting in a hazy sky with halos around the light sources.
                                                # When set to 0, there is no contribution (default: 0.0)
        rayleigh_scattering_scale=0.0331        # Controls interaction of light with small particles like air molecules.
                                                # Dependent on light wavelength, resulting in a blue sky in the day or
                                                # red sky in the evening (default: 0.0331)
    )
    weather_dict = {
        "cloudiness": cloudiness,
        "precipitation": precipitation,
        "precipitation_deposits": precipitation,
        "wind_intensity": wind_intensity,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": sun_altitude_angle,
        "fog_density": fog_density,
        "fog_distance": 15.0,
        "wetness": 0.0,
        "fog_falloff": 0.0,
        "scattering_intensity": 0.0,
        "mie_scattering_scale": 0.0,
        "rayleigh_scattering_scale": 0.0331,
    }

    return a_weather, weather_dict


def put_elements_in_bins(elements, num_of_bin, min_value, max_value):
    y = [0 for _ in range(num_of_bin)]
    bin_step = (max_value - min_value) / num_of_bin
    x = [min_value + bin_step/2 + bin_step*i for i in range(0, num_of_bin)]
    limits = [min_value + bin_step*i for i in range(1, num_of_bin)]
    for el in elements:
        for i, l in enumerate(limits):
            if el < l:
                y[i] += 1
                break
    return x, y


def print_betavariate():
    """
    mean = a / (a + b)
    """
    import matplotlib.pyplot as plt
    elements = []
    for i in range(100000):
        elements.append(random.uniform(-90, 90))  # random.betavariate(alpha=2, beta=2) * 100)
    x, y = put_elements_in_bins(elements, 20, min(elements), max(elements))
    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    print_betavariate()
