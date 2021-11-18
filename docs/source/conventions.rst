============
Conventions
============

Some conventions that have been implemented:

    * All metadata names are lower case and `_` separated.
	
Survey Names
----------------

Survey names are not standardized yet, but will be in the future.  Survey names are commonly a string of up to 10 characters, though there is no limit. Characters representing the project or geographic location are commonly used.  For example a survey in Long Valley could be **lv**, a survey of the continental US could be **conus**, a NSF funded project on imaging magma under Mount St. Helens (iMUSH) could be **imush**.   

	
Station Names
----------------

Stations names are not standardized yet, but will be in the future.  Station names are often 4-6 characters. Convention is a 2 or 3 character survey representation followed by a 2-4 digit number representing the station number.  

For example a survey in California could begin with **ca** and station 1 would be **ca001**.  

Now that 3-D grids are more common people name lines or rows with a letter or number.  For instance the EarthScope/MT Array across the US uses {state}{line}{number}.  So a station could be can10 for station 10 on the N easting line in California.

Future suggestion is:

    {survey_name}{line}{row}{number} 
	
Run Names
-----------

Run names have not been standardized yet but will be in the future.  There are 2 conventions, one to use alphabetic run names like run **a** and the other is numeric like run **001**.

The disadvantage of alphabetic names is they can be limited by the alphabet if the number of runs gets large.  Alphabetic names are common in long period experiments where you have a handful of runs when you need to change batteries or fix cables.

Numeric names are more flexible and will probably be the standard in the future.  For broadband experiments where multiple sampling rates are used there can be lots of runs.  For example if you have a continuous band sampling at 50 samples/second with 5 second bursts of 1000 samples/second every minute then you are going to have 60 short runs per hour. Keeping track of these can be tedious so a numeric counter would be the easiest.  

Future suggestions would be a 4 digit string, maybe include the sample rate which could be a character similar to seismic FDSN standards.  

    {sample_rate}{run_number}  


Channel Names
---------------	

Channel names have not been standardized yet but will be in the future.  Direction indicators are often given as:

    * **x**: northing direction or strike direction
    * **y**: easting direction or across strike direction
    * **z**: vertical direction	

For MT we have 2 main types of channels `electric` and `magnetic`. 

`Electric` channel names often start with an **e** and are followed by a directional indicator, like **ex** for an electric channel in the northing direction.  

`Magnetic` channel names often start with an **h** or **b** followed by a directional indicator, like **hz** for a vertical magnetic channel.

Extending this to other data types we have `Auxiliary` channels which could be any other type of geophysical measurement, like temperature or battery voltage.  Suggest using the full name at the moment, maybe in the future we will have measurment codes like seismic FDSN, but they can be cryptic.  

In the future channel names will likely be standardized as:

    `{channel_type}{channel_direction}{channel_number}`
	
to allow for flexibility to other methods like IP and DC surveys.



    