from strenum import StrEnum


class SessionFields(StrEnum):
    ID = "evse_uid"
    TYPE = "type"
    STARTED = "started_at"
    STOPPED = "stopped_at"
    LAST_ENERGY_CHANGE = "last_energy_change_at"
    ENERGY_WH = "energy_wh"
    FULL_SESSION_DURATION = "duration_seconds"
    STATUS = "status"
    LOCATION = "location_name"
    CITY = "location_city"
    COUNTRY = "country"

    # Encoded
    ID_ENCODED = f'{ID}_encoded'
    LOCATION_ENCODED = f'{LOCATION}_encoded'


class TrainingFields(StrEnum):
    ENERGY_KWH = "energy_kwh"
    CHARGING_DURATION = "session_charging_duration"
    STATION_AVG_POWER = "station_avg_power"
    POINT_AVG_POWER = "point_avg_power"
    STATION_POINTS = "station_charging_points"
    START_DATE = "session_date"
    END_DATE = "end_date"
    OCCUPANCY = "occupancy"
    DATE = "date"
    FIRST_DAY = 'first_active_day'
    FINAL_DAY = 'final_active_day'
    LAST_SESSION_PRE_DATA = 'last_session_pre_data'
    WEEKDAY = "weekday"
    MONTH = "month"
    QUARTER = "quarter"
    SEASON = "season"
    YEAR = "year"
    HOLIDAY = "holiday_period"
    POPULATION = "population"
    CITY_TYPE = "city_type"
    SESSION_DURATION = "session_duration"
    DAY_PERIOD = "day_period"
    PERIOD_DURATION = "period_duration"
    ACTIVE_TIME = "usage_time"
    AVERAGE_PERIOD_BUSYNESS = "avg_day_period_busyness"
    AVERAGE_BUSYNESS = 'avg_total_busyness'
    DAYS_SINCE_START = 'days_since_data_start'
    WEEKS_SINCE_START = 'weeks_since_data_start'
    DAYS_SINCE_POINT_ACTIVE = 'days_since_point_active'
    WEEKS_SINCE_POINT_ACTIVE = 'weeks_since_point_active'
    GAP_DAYS = 'gap_days'
    DAILY_PERIOD_BUSYNESS = 'daily_period_busyness'
    DAILY_PERIOD_BUSYNESS_CAT = 'daily_period_busyness_cat'
    DAILY_BUSYNESS = 'daily_busyness'
    ROLLING_AVG_1D = 'daily_rolling_avg_1d'
    ROLLING_AVG_7D = 'daily_rolling_avg_7d'
    ROLLING_AVG_30D = 'daily_rolling_avg_30d'
    ROLLING_PERIOD_AVG_1D = 'daily_period_rolling_avg_1d'
    ROLLING_PERIOD_AVG_7D = 'daily_period_rolling_avg_7d'
    ROLLING_PERIOD_AVG_30D = 'daily_period_rolling_avg_30d'
    ROLLING_PERIOD_IQR_7D = 'daily_period_rolling_IQR_7d'
    ROLLING_PERIOD_IQR_30D = 'daily_period_rolling_IQR_30d'
    PAST_WEEK_SAME_PERIOD = 'past_week_same_period'
    WEEKLY_CUMULATIVE_AVG = 'weekly_cumulative_average'
    DAILY_CUMULATIVE_AVG = 'daily_cumulative_average'
    PERIOD_WEEKLY_CUMULATIVE_AVG = 'period_weekly_cumulative_average'
    PERIOD_DAILY_CUMULATIVE_AVG = 'period_daily_cumulative_average'

    # Encoded
    CITY_TYPE_ENCODED = f'{CITY_TYPE}_encoded'
    DAY_PERIOD_ENCODED = f'{DAY_PERIOD}_encoded'
    WEEKDAY_ENCODED = f'{WEEKDAY}_encoded'
    MONTH_ENCODED = f'{MONTH}_encoded'
    SEASON_ENCODED = f'{SEASON}_encoded'

    # Interacting features
    WEEKDAY_ENC_1D_SHIFT = 'weekday_enc_1d_shift'
    HOLIDAY_1D_SHIFT = 'holiday_1d_shift'
    DAILY_BUSYNESS_2D_SHIFT = 'daily_busyness_2d_shift'
    DAILY_PERIOD_BUSYNESS_2D_SHIFT = 'daily_period_busyness_2d_shift'

    WEEKDAY_HOLIDAY = 'weekday_x_holiday'
    WEEKDAY_PERIOD = 'weekday_x_day_period'
    WEEKDAY_HOLIDAY_1D_SHIFT = 'weekday_x_holiday_1d_shift'

    DAILY_BUSYNESS_CHANGE = 'busyness_change'
    DAILY_PERIOD_BUSYNESS_CHANGE = 'period_busyness_change'
