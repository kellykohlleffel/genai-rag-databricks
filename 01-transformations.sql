/** Create each winery and vineyard review as a single field vs multiple fields **/
CREATE OR REPLACE TABLE vineyard_data_single_string AS
   SELECT WINERY_OR_VINEYARD, CONCAT(
       'The winery name is ', IFNULL(WINERY_OR_VINEYARD, ' Name is not known'), '.',
       ' Wine region: ', IFNULL(CA_WINE_REGION, 'unknown'),
       ' The AVA Appellation is the ', IFNULL(AVA_APPELLATION_SUB_APPELLATION, 'unknown'), '.',
       ' The website associated with the winery is ', IFNULL(WEBSITE, 'unknown'), '.',
       ' The price range is ', IFNULL(PRICE_RANGE, 'unknown'), '.',
       ' Tasting Room Hours: ', IFNULL(TASTING_ROOM_HOURS, 'unknown'), '.',
       ' The reservation requirement is: ', IFNULL(RESERVATION_REQUIRED, 'unknown'), '.',
       ' Here is a complete description of the winery or vineyard: ', IFNULL(WINERY_DESCRIPTION, 'unknown'), '.',
       ' The primary varietal this winery offers is ', IFNULL(PRIMARY_VARIETALS, 'unknown'), '.',
       ' Thoughts on the Tasting Room Experience: ', IFNULL(TASTING_ROOM_EXPERIENCE, 'unknown'), '.',
       ' Amenities: ', IFNULL(AMENITIES, 'unknown'), '.',
       ' Awards and Accolades: ', IFNULL(AWARDS_AND_ACCOLADES, 'unknown'), '.',
       ' Distance Travel Time considerations: ', IFNULL(DISTANCE_AND_TRAVEL_TIME, 'unknown'), '.',
       ' User Rating: ', IFNULL(USER_RATING, 'unknown'), '.',
       ' The secondary varietal for this winery is: ', IFNULL(SECONDARY_VARIETALS, 'unknown'), '.',
       ' Wine Styles for this winery are: ', IFNULL(WINE_STYLES, 'unknown'), '.',
       ' Events and Activities: ', IFNULL(EVENTS_AND_ACTIVITIES, 'unknown'), '.',
       ' Sustainability Practices: ', IFNULL(SUSTAINABILITY_PRACTICES, 'unknown'), '.',
       ' Social Media Channels: ', IFNULL(SOCIAL_MEDIA, 'unknown'), '.',
       ' The address is ',
           IFNULL(ADDRESS, 'unknown'), ', ',
           IFNULL(CITY, 'unknown'), ', ',
           IFNULL(STATE, 'unknown'), ', ',
           IFNULL(ZIP, 'unknown'), '.',
       ' The Phone Number is ', IFNULL(PHONE, 'unknown'), '.',
       ' Winemaker: ', IFNULL(WINEMAKER, 'unknown'),
       ' Did Kelly Kohlleffel recommend this winery?: ', IFNULL(KELLY_KOHLLEFFEL_RECOMMENDED, 'unknown')
   ) AS winery_information
   FROM california_wine_country_visits;

SELECT * FROM vineyard_data_single_string;
