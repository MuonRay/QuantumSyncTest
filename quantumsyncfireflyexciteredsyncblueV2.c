/* -----------------------------------------------------------------------
 * Title:    quantumsyncfirefly.c
 * Author:   J. Campbell
 * 
 * #this version gives excited state as blue
 * 
 * #use the other version to give excited state as red
 * 
 * # let sync for superposition, and measure to collapse the sync-wavefunction to induce the separability of the 2 states correcponding to the 2 nodes
 * 
 * # this is an example of a scalable operational quantum protocol, taken from quantum many-body entanglement and applied in a network of Kuramoto oscillators scenario in active Mermin
 * type device hardware
 *           
 * Date:     06/09/2015
 * Hardware: ATtiny85
 * Reference: https://www.youtube.com/watch?v=2d6dHHhKOOE
 * Description:
 * This tiny program simulates a single quantum firefly. It uses an RGB-LED to flash
 * once every two seconds. If put together with other fireflies, it uses a
 * light sensor (a photo transistor) to detect flashes from other fireflies
 * nearby. The program tries to synchronize its own flash with the flash of
 * its neighbours.
 * 
 * This Program is written in pure C and so may not be as easy to read as conventional Arduino programs
 * Moreover in certain hardware it may cause an effect of "Bricking", that is to say the bootloader may not read
 * the device on a COM Port anymore. The best way to rectify this is to create a hard reset on the device and upload the 
 * default sketch file to the device to wipe the chip clean of the original code. (has to be done within 8 seconds on Arduino Pro-Micro, other boards may vary)
 * 2015 Muon Ray Enterprises
 * 
 * Chip type: ATtiny85
 * Clock frequency: Default internal clock 8MHz, choose 1MHz 
 * So clock can be set as 8MHz/4 = 2MHz max for each pin. 
 * (note: it is possible to set at 16MHz but is overkill for this application)
 * 
 *
 * The RGB-LED is a commond cathode type.
 *
 *   Vcc   o     ATtiny85               o Vcc (+5V)
 *         |    +--------v--------+     | 
 *        100k  |                 |     |
 *         +----+ 1 PB5     Vcc 8 +-----+
 *              |                 |
 *         +----+ 2 PB3     PB2 7 +-- 100R --|>|--+  Green
 *  phototrans. |                 |               |
 *         +----+ 3 PB4     PB1 6 +-- 100R --|>|--+  Red
 *         R4   |                 |               |
 *         +----+ 4 GND     PB0 5 +-- 100R --|>|--+  Blue
 *         |    |                 |               |
 *         |    +-----------------+               |
 *         |                                      |
 *        _|_                                    _|_
 *
 * R4 has to be adjusted to the used LDR (1k-50k).
 * If you use a phototransistor, try higher values (>=100k). 
 *
 */
#include <Arduino.h>
#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

#define FLASH_POWER 8000      // power level at which the firefly flashes
#define POWER_BOOST 400       // amount of power to add, for every other flash
#define FLASH_DELAY 200       // how long lasts the flash
#define DAYLIGHT 240          // values higher than that are recognized as daylight
#define DAYLIGHT_DELAY 10000  // wait 10 seconds, if daylight detected
#define BLIND_AFTER_OTHER 800 // how long are we blind after another flash
#define BLIND_AFTER_SELF 100  // how long are we blind after our own flash
#define THRESHOLD_DELTA 20    // added to the ambient light value


// #define NEW_RGB // use this to choose different leds pins

#ifdef NEW_RGB
#define B_BIT 0               // pin 5  -- LED pin 2
#define R_BIT 1               // pin 6  -- LED pin 4
#define G_BIT 2               // pin 7  -- LED pin 1
#else
#define R_BIT 0               // pin 5  -- LED pin 2
#define B_BIT 1               // pin 6  -- LED pin 4
#define G_BIT 2               // pin 7  -- LED pin 1
#endif


static volatile uint8_t act_light = 0;  // value for lightness
static uint8_t softscale = 0; // used to prescale the timer 
static volatile uint8_t r = 0;         // rgb, set these to control the color
static volatile uint8_t g = 0;         // green
static volatile uint8_t b = 0;         // blue
static uint8_t rtmp = 0;      // tmp vars for rgb, do not use directly
static uint8_t gtmp = 0;      
static uint8_t btmp = 0;      



/* -----------------------------------------------------
 * ADC interrupt
 * The AD conversion is in free running mode.
 * -> prescaler 128 -> 75.000 samples at 9.600.000 Hz
 */
SIGNAL(ADC_vect) {
  act_light = ADCH;           // read only 8-bit
}



/* -----------------------------------------------------
 * Timer0 overflow interrupt
 * F_CPU 9.600.000 Hz 
 * -> prescaler 0, overrun 256 -> 37.500 Hz 
 * -> 256 steps -> 146 Hz
 *
 * Read global rgb and save it to tmp vars and decide, to switch
 * on R, G or B. 
 * Range is for rgb is 0-255 (dark to full power)
 */
SIGNAL(TIM0_OVF_vect) {	
  // every 256th step take over new values
  if (++softscale == 0) {    
    rtmp = r;
    gtmp = g;
    btmp = b;
    // check if switch on r, g and b
    if (rtmp > 0) {        
      PORTB |= (1 << R_BIT);
    } 
    if (gtmp > 0) {
      PORTB |= (1 << G_BIT);
    } 
    if (btmp > 0) {
      PORTB |= (1 << B_BIT);
    } 
  }
  // check if switch off r, g and b
  if (softscale == rtmp) {     
    PORTB &= ~(1 << R_BIT);
  }
  if (softscale == gtmp) {
    PORTB &= ~(1 << G_BIT);
  }
  if (softscale == btmp) {
    PORTB &= ~(1 << B_BIT);
  }
}



/* -----------------------------------------------------
 * Converts a hue into rgb values (HSV -> RGB). We are using
 * the HSV model, because there the color is directly related
 * to a single value, the hue and not three values as in RGB.
 * That makes it really easy to change the color from red to blue.
 * Even a full cycle (all rainbow colors) would be easy. You would
 * just increment this hue and don't have to deal with R, G and B.
 *
 * You can take a look at http://en.wikipedia.org/wiki/HSL_and_HSV
 * for how the HSV color model works. 
 * 
 * Normally the HSV model needs a full circle (360�) to distribute
 * all rainbow colors over this circle. But 360 is too big for a
 * single byte. So we use a small trick here. We DEFINE simply,
 * that our circle has only 252�, not 360�. Also 252 is dividable 
 * by 6 and in the HSV model we have 6 segments. In every segment
 * two values of R, G and B stays constant (0 or 252) and another
 * value ramps up or down.
 *
 * Here is a small table with all colors. Segments stands for
 * the color segment. Color, ok that's obvious. Hue is the 
 * hue from 0� to 360�. Hue(252) stands for the hue, that we are
 * using, from 0� to 252�.
 *
 * Segment - Color     - hue - hue(252)
 *       0 - red       -   0 -   0
 *       1 - yellow    -  60 -  42
 *       2 - green     - 120 -  84
 *       3 - turquoise - 180 - 126
 *       4 - blue      - 240 - 168
 *       5 - pink      - 300 - 210
 *       0 - red       - 360 - 252
 */


//modified so that when excited flashes red and trends towards blue when synced


void h_to_rgb(uint8_t hue) {
  uint8_t hd = hue / 42;      // 42 == 252/6, determines the segment this hue lies in
  uint8_t hi = hd % 6;        // gives 0-5, to be sure even if hd was greater than 5.
                              // Our segments are 0-5.
  uint8_t f = hue % 42;       // Calculates the hue within the segment. 
  uint8_t fs = f * 6;         // We need 0..252 for our color, so we multiply by 6.
  
  switch (hi) {
  case 0: // Red to yellow
    r = 252;     // Red: full power
    g = fs;      // Green: ramp up
    b = 0;       // Blue: off
    break;
  case 1: // Yellow to green
    r = 252 - fs;  // Red: ramp down
    g = 252;     // Green: full power
    b = 0;       // Blue: off
    break;
  case 2: // Green to cyan
    r = 0;       // Red: off
    g = 252;     // Green: full power
    b = fs;      // Blue: ramp up
    break;
  case 3: // Cyan to blue
    r = 0;       // Red: off
    g = 252 - fs;  // Green: ramp down
    b = 252;     // Blue: full power
    break;
  case 4: // Blue to pink
    r = fs;      // Red: ramp up
    g = 0;       // Green: off
    b = 252;     // Blue: full power
    break;
  case 5: // Pink to red
    r = 252;     // Red: full power
    g = 0;       // Green: off
    b = 252 - fs;  // Blue: ramp down
    break;
  }
}

// The rest of the code remains unchanged





int main(void) {	
	
  uint8_t i = 0;              // used for loops
  uint8_t light = 0;          // current lightness
  uint8_t nervous = 0;        // value for how nervous we are. 0 to 168. 
  uint16_t threshold = 0;     // threshold to determine a flash
  uint16_t power = 0;         // current power level
  uint16_t blind = 0;         // stores for how many cycles we are blind
  uint16_t flash_power = FLASH_POWER;  // if power is > flash_power, then flash
	
  // enable pins as output
  DDRB |= 
    (1 << PB3) |              // PB3 acts as power supply for photo transistor voltage divider
    (1 << R_BIT) |            // R
    (1 << G_BIT) |            // G
    (1 << B_BIT);             // B

  // power on the voltage divider
  PORTB |= (1 << PB3);	
			
  // timer 0 setup, prescaler none
  TCCR0B |= (0 << CS02) | (0 << CS01) | (1 << CS00);

  // enable timer 0 interrupt
  TIMSK |= (1 << TOIE0);	

  // enable adc
  ADCSRA |= 
    (1 << ADEN) |             // enable ADC
    (1 << ADATE) |            // enable auto triggering
    (1 << ADIE) |             // enable ADC interrupt
    (1 << ADSC) |             // start conversion
    (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);  // prescaler 128 --> 75k samples at 9.6 MHz

  // select ADC channel
  ADMUX |= 
    (0 << REFS0) |            // select Vcc as reference
    (1 << ADLAR) |            // ADC output as 8-bit, left adjusted
    (1 << MUX1);              // select channel 2, pin 3

  // enable all interrupts
  sei();

  // intro, blink red 5 times
  for (i = 0; i < 5; i++) {
    r = 255;
    _delay_ms(100);
    r = 0;
    _delay_ms(100);
  }


  // compute threshold of the ambient light
  for (i = 0; i < 4; i++) {
    threshold += act_light;
    _delay_ms(500);
  }	
  threshold = threshold >> 2;       // divides by 4
  threshold += THRESHOLD_DELTA;     // move the threshold above the ambient average

  // try to sleep some randomized time
  i = (act_light & 0x03);           // use the last (right most) 2 bits of the actual lightness
  while (i--) {
    _delay_ms(1000);
  }

  // enter the main loop
  while (1) {

    _delay_us(500);                 // every cylce takes at least 0.5 ms

    if (power > 6000) {             // increase the power level with a, first fast ascending,
      power += 1;                   // later slower ascending, function
    }
    else if (power > 4000) {
      power += 2;
    }
    else if (power > 3000) {
      power += 4;
    }
    else if (power > 2000) {
      power += 8;
    }
    else {
      power += 16;
    }

    light = act_light;              // read the actual lightness
    if (!blind) {                   
      if (light > DAYLIGHT) {	    // if we detect daylight, then it's too bright to work
	g = 32;                     // switch color to green
        _delay_ms(DAYLIGHT_DELAY);  // and wait 10 seconds
	g = 0;                      
      }
      else if (light > threshold) { // was it a flash?
	// if the flash comes in, when we are in between 2000 and 7000, then we detect
	// the flash as not in sync and increase the nervous level ...
	if ((power > 2000) && (power < 7000)) {
	  nervous = (nervous >= 158) ? 168 : (nervous + 10);
	}
	else {
	  if (nervous > 5) {        // ... otherwise decrease it
	    nervous -= 5;
	  }
	}
        power += POWER_BOOST;       // boost the power
        blind = BLIND_AFTER_OTHER;  // and we are blind for the next cycles
      }
    }
    else if (blind > 0) {           // if we are blind, then do nothing
      blind--;							
    }
						
    if (power > flash_power) {      // if there is enough power, then we flash
      h_to_rgb(168 - nervous);      // display the color, depending on the nervous level
      _delay_ms(FLASH_DELAY);       // wait
      r = 0;                        // flash off
      g = 0;
      b = 0;                       
      power = 0;                    // reset power
      blind = BLIND_AFTER_SELF;     // blind after our own flash for some cycles
      if (nervous > 3) {            // decrease the nervous level
	nervous -= 3;
      }
    }
		
  }

  return 0;
	
}
