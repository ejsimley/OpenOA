import importlib
import itertools

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point


def wrap_180(x):
    """
    Converts an angle or array of angles in degrees to the range -180 to +180 degrees.

    Args:
        x (:obj:`float` or :obj:`numpy.ndarray`): Input angle(s) (degrees)

    Returns:
        :obj:`float` or :obj:`numpy.ndarray`: The input angle(s) converted to the range -180 to +180 degrees (degrees)
    """
    input_type = type(x)

    x = x % 360.0  # convert to range 0 to 360 degrees
    x = np.where(x > 180.0, x - 360.0, x)
    return x if input_type == np.ndarray else float(x)


class AssetData(object):
    """
    This class wraps around a Pandas dataframe that contains
    metadata about the plant assets. It provides some useful functions
    to work with this data (e.g., calculating nearest neighbors, etc.).
    """

    def __init__(self, engine="pandas"):
        self._asset = None
        self._nearest_neighbors = None
        self._nearest_towers = None
        self._engine = engine
        if engine == "spark":
            self._sql = importlib.import_module("pyspark.sql")
            self._pyspark = importlib.import_module("pyspark")
            self._sc = self._pyspark.SparkContext.getOrCreate()
            self._sqlContext = self._sql.SQLContext.getOrCreate(self._sc)

    def load(self, path, name, format="csv"):
        if self._engine == "pandas":
            self._asset = pd.read_csv("%s/%s.%s" % (path, name, format))
        elif self._engine == "spark":
            self._asset = (
                self._sqlContext.read.format("com.databricks.spark.csv")
                .options(header="true", inferschema="true")
                .load("%s/%s.csv" % (path, name))
                .toPandas()
            )

    def save(self, path, name, format="csv"):
        if self._engine == "pandas":
            self._asset.to_csv("%s/%s.%s" % (path, name, format))
        elif self._engine == "spark":
            self._sqlContext.createDataFrame(self._asset).write.mode("overwrite").format(
                "com.databricks.spark.csv"
            ).options(header="true", inferschema="true").save("%s/%s.csv" % (path, name))

    def prepare(self, active_turbine_ids=None, active_tower_ids=None, srs="epsg:4326"):
        """Prepare the asset data frame for further analysis work. Currently, this function calls parse_geometry(srs)
        and calculate_nearest(active_turbine, active_tower), passing through the arguments to this function.

        Args:
            active_turbine_ids (:obj:`list`, optional): Optional list of IDs of turbines to consider. If None, all
                turbines will be considered. Defaults to None.
            active_tower_ids (:obj:`list`, optional): Optional list of IDs of met towers to consider. If None, all met
                towers will be considered. Defaults to None.
            srs (:obj:`str`, optional): Used to define the coordinate
                reference system (CRS). Defaults to the European
                Petroleum Survey Group (EPSG) code 4326 to be used with
                the World Geodetic System reference system, WGS 84.

        Returns: None
            Sets asset 'geometry', 'nearest_turbine_id' and 'nearest_tower_id' column.

        """
        self.parse_geometry(srs)
        self.calculate_nearest(
            active_turbine_ids=active_turbine_ids, active_tower_ids=active_tower_ids
        )

    def parse_geometry(self, srs="epsg:4326", zone=None, longitude=None):
        """Calculate UTM coordinates from latitude/longitude.

        The UTM system divides the Earth into 60 zones, each 6deg of
        longitude in width. Zone 1 covers longitude 180deg to 174deg W;
        zone numbering increases eastward to zone 60, which covers
        longitude 174deg E to 180deg. The polar regions south of 80deg S
        and north of 84deg N are excluded.

        Ref: http://geopandas.org/projections.html

        Args:
            srs (:obj:`str`, optional): Used to define the coordinate
                reference system (CRS). Defaults to the European
                Petroleum Survey Group (EPSG) code 4326 to be used with
                the World Geodetic System reference system, WGS 84.
            zone (:obj:`int`, optional): UTM zone. If set to None
                (default), then calculated from the longitude.
            longitude (:obj:`float`, optional): Reference longitude for
                calculating the UTM zone. If None (default), then taken
                as the average longitude of all assets.

        Returns: None
            Sets asset 'geometry' column.
        """
        if zone is None:
            # calculate zone
            if longitude is None:
                longitude = self.df["longitude"].mean()
            zone = int(np.floor((180 + longitude) / 6.0)) + 1

        to_crs = f"+proj=utm +zone={zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        transformer = Transformer.from_crs(srs.upper(), to_crs)
        lats, lons = transformer.transform(
            self._asset["latitude"].values, self._asset["longitude"].values
        )
        self._asset["geometry"] = [Point(lat, lon) for lat, lon in zip(lats, lons)]

    def calculate_nearest(self, active_turbine_ids=None, active_tower_ids=None):
        """Create or overwrite a column called 'nearest_turbine_id' or 'nearest_tower_id' which contains the asset id
        of the closest active turbine or tower to the closest turbine or tower. If specified, the column will only be
        valid for the turbines or towers listed in the arguments of this function. Additionally, it will only calculate
        the value of the correct column for each asset. Turbines, for example, will have null 'nearest_tower_id' and
        vice versa.

        Args:
            active_turbine_ids (:obj:`list`, optional): Optional list of IDs of turbines to consider. If None, all
                turbines will be considered. Defaults to None.
            active_tower_ids (:obj:`list`, optional): Optional list of IDs of met towers to consider. If None, all met
                towers will be considered. Defaults to None.

        Returns: None
            Sets asset 'nearest_turbine_id' and 'nearest_tower_id' column.
        """
        if active_turbine_ids is None:
            active_turbine_ids = self._asset.loc[self._asset["type"] == "turbine", "id"].tolist()

        if active_tower_ids is None:
            active_tower_ids = self._asset.loc[self._asset["type"] == "tower", "id"].tolist()

        self._asset["nearest_turbine_id"] = None
        if active_turbine_ids is not None and len(active_turbine_ids) > 0:
            nn = self.nearest_neighbors()
            for k, v in nn.items():
                v = [val for val in v if val in active_turbine_ids]
                self._asset.loc[self._asset["id"] == k, "nearest_turbine_id"] = v[0]
        if active_tower_ids is not None and len(active_tower_ids) > 0:
            nt = self.nearest_towers()
            self._asset["nearest_tower_id"] = None
            for k, v in nt.items():
                v = [val for val in v if val in active_tower_ids]
                self._asset.loc[self._asset["id"] == k, "nearest_tower_id"] = v[0]

    def distance_matrix(self, asset_type=None):
        """
        Returns a matrix containing distances between each pair of assets, for all assets or a specific asset type.

        Args:
            asset_type (:obj:`string`, optional): Optional asset type to calculate distances for
            ("turbine" or "tower"). If None, all assets are included. Defaults to None.

        Returns:
            :obj:`numpy.ndarray`: Matrix containing distances between each pair of assets.
        """
        if asset_type is None:
            df_asset_sub = self._asset
        else:
            df_asset_sub = self._asset.loc[self._asset["type"] == asset_type].reset_index()
        ret = np.ones((df_asset_sub.shape[0], df_asset_sub.shape[0])) * -1
        for i, j in itertools.combinations(df_asset_sub.index, 2):
            point1 = df_asset_sub.loc[i, "geometry"]
            point2 = df_asset_sub.loc[j, "geometry"]
            distance = point1.distance(point2)
            ret[i, j] = ret[j, i] = distance
        return ret

    def direction_matrix(self, asset_type=None):
        """
        Returns a matrix containing directions between each pair of assets, for all assets or a specific asset type.

        Args:
            asset_type (:obj:`string`, optional): Optional asset type to calculate directions for
                ("turbine" or "tower"). If None, all assets are included. Defaults to None.

        Returns:
            :obj:`numpy.ndarray`: Matrix containing directions between each pair of assets (defined as the direction
                from the asset given by the 1st index to the asset given by the 2nd index, relative to north)
        """
        if asset_type is None:
            df_asset_sub = self._asset
        else:
            df_asset_sub = self._asset.loc[self._asset["type"] == asset_type].reset_index()
        ret = np.ones((df_asset_sub.shape[0], df_asset_sub.shape[0])) * -1
        for i, j in itertools.permutations(df_asset_sub.index, 2):
            point1 = df_asset_sub.loc[i, "geometry"]
            point2 = df_asset_sub.loc[j, "geometry"]
            direction = np.degrees(np.arctan2(point2.x - point1.x, point2.y - point1.y)) % 360.0
            ret[i, j] = direction
        return ret

    def get_freestream_turbines(self, wd, freestream_method="sector", sector_width=45.0):
        """
        Returns a list of freestream (unwaked) turbines for a given wind direction. Freestream turbines can be
        identified using different methods ("sector" or "IEC" methods). For the sector method, if there are any
        turbines upstream of a turbine within a fixed wind direction sector centered on the wind direction of interest,
        defined by the sector_width argument, the turbine is condiered waked. The IEC method uses the freestream
        definition provided in Annex A of IEC 61400-12-1 (2005).

        Args:
            wd (:obj:`float`): Wind direction to identify freestream turbines for (degrees)
            freestream_method (:obj:`string`, optional): Method used to identify freestream turbines
                ("sector" or "IEC"). Defaults to "sector".
            sector_width (:obj:`float`, optional): Width of wind direction sector centered on the wind direction of
                interest used to determine whether a turbine is waked for the "sector" method (degrees). For a given
                turbine, if any other upstream turbines are located within the sector, then the turbine is considered
                waked. Defaults to 45 degrees.

        Returns:
            :obj:`list`: List of freestream turbine asset IDs
        """
        turbine_direction_matrix = self.direction_matrix(asset_type="turbine")

        if freestream_method == "sector":
            # find turbines for which no other upstream turbines are within half of the sector width of the specified
            # wind direction
            freestream_indices = np.all(
                (np.abs(wrap_180(wd - turbine_direction_matrix)) > 0.5 * sector_width)
                | np.diag(np.ones(len(turbine_direction_matrix), dtype=bool)),
                axis=1,
            )
        if freestream_method == "IEC":
            # find freestream turbines according to the definition in Annex A of IEC 61400-12-1 (2005)
            turbine_distance_matrix = self.distance_matrix(asset_type="turbine")

            # normalize distances by rotor diameters of upstream turbines
            rotor_diameters = np.ones((len(turbine_direction_matrix), 1)) * np.array(
                self._asset.loc[self._asset["type"] == "turbine", "rotor_diameter_m"]
            )
            turbine_distance_matrix /= rotor_diameters

            freestream_indices = np.all(
                (
                    (turbine_distance_matrix > 2)
                    & (
                        np.abs(wrap_180(wd - turbine_direction_matrix))
                        > 0.5
                        * (1.3 * np.degrees(np.arctan(2.5 / turbine_distance_matrix + 0.15)) + 10)
                    )
                )
                | (turbine_distance_matrix > 20)
                | (turbine_distance_matrix < 0),
                axis=1,
            )
        else:
            raise ValueError(
                'Invalid freestream method. Currently, "sector" and "IEC" are supported.'
            )

        return self.turbine_ids()[freestream_indices]

    def asset_ids(self):
        return self._asset.loc[:, "id"].values

    def tower_ids(self):
        return self._asset.loc[self._asset["type"] == "tower", "id"].values

    def turbine_ids(self):
        return self._asset.loc[self._asset["type"] == "turbine", "id"].values

    def remove_assets(self, to_delete):
        self._asset = self._asset.loc[~self._asset["id"].isin(to_delete), :].reset_index(drop=True)

    def nearest_neighbors(self):
        if self._nearest_neighbors is not None:
            return self._nearest_neighbors

        ret = {}
        towers = self._asset.loc[self._asset["type"] == "tower", :].index
        turbines = self._asset.loc[self._asset["type"] == "turbine", :].index
        m = self.distance_matrix()
        for i in turbines:
            row = m[i]
            row[row == -1] = float("inf")
            row[towers.tolist()] = float("inf")
            ret[self._asset.loc[i, "id"]] = [self._asset.loc[x, "id"] for x in row.argsort()]

        self._nearest_neighbors = ret
        return ret

    def nearest_tower_to(self, id):
        return self._asset.loc[self._asset["id"] == id, "nearest_tower_id"].values[0]

    def nearest_turbine_to(self, id):
        return self._asset.loc[self._asset["id"] == id, "nearest_turbine_id"].values[0]

    def nearest_towers(self):
        if self._nearest_towers is not None:
            return self._nearest_towers

        ret = {}
        turbines = self._asset.loc[self._asset["type"] == "turbine", :].index
        m = self.distance_matrix()
        for i in turbines:
            row = m[i]
            row[row == -1] = float("inf")
            row[turbines.tolist()] = float("inf")
            ret[self._asset.loc[i, "id"]] = [self._asset.loc[x, "id"] for x in row.argsort()]

        self._nearest_towers = ret
        return ret

    def rename_columns(self, mapping):
        for k in list(mapping.keys()):
            if k != mapping[k]:
                self._asset[k] = self._asset[mapping[k]]
                self._asset[mapping[k]] = None

    def head(self):
        return self._asset.head()

    @property
    def df(self):
        return self._asset

    @df.setter
    def df(self, value):
        self._asset = value
