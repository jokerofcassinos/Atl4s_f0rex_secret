//+------------------------------------------------------------------+
//|                                           Atl4sDataExporter.mq5 |
//|                                    Atl4s-Forex Data Export Tool |
//|                          Exports M1 data for Python Backtesting |
//+------------------------------------------------------------------+
#property copyright "Atl4s-Forex"
#property version   "1.00"
#property script_show_inputs

//--- Input Parameters
input string   ExportSymbol = "EURUSD";        // Symbol to export
input int      DaysToExport = 30;              // Days of history to export
input string   ExportPath = "Atl4s_Export";    // Folder name in MQL5/Files

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("=== ATL4S DATA EXPORTER v1.0 ===");
   Print("Symbol: ", ExportSymbol);
   Print("Days: ", DaysToExport);
   
   // Calculate date range
   datetime endDate = TimeCurrent();
   datetime startDate = endDate - (DaysToExport * 24 * 60 * 60);
   
   Print("Exporting from ", TimeToString(startDate), " to ", TimeToString(endDate));
   
   // Request M1 data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates(ExportSymbol, PERIOD_M1, startDate, endDate, rates);
   
   if(copied <= 0)
   {
      Print("ERROR: Failed to copy rates. Error: ", GetLastError());
      return;
   }
   
   Print("Copied ", copied, " M1 candles.");
   
   // Create filename
   string filename = ExportPath + "/" + ExportSymbol + "_M1.csv";
   
   // Open file
   int fileHandle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   
   if(fileHandle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot create file. Error: ", GetLastError());
      return;
   }
   
   // Write header
   FileWrite(fileHandle, "time", "open", "high", "low", "close", "volume");
   
   // Write data (oldest first)
   for(int i = copied - 1; i >= 0; i--)
   {
      string timeStr = TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES | TIME_SECONDS);
      
      FileWrite(fileHandle, 
         timeStr,
         DoubleToString(rates[i].open, 5),
         DoubleToString(rates[i].high, 5),
         DoubleToString(rates[i].low, 5),
         DoubleToString(rates[i].close, 5),
         IntegerToString(rates[i].tick_volume)
      );
   }
   
   FileClose(fileHandle);
   
   Print("=== EXPORT COMPLETE ===");
   Print("File saved to: MQL5/Files/", filename);
   Print("Candles exported: ", copied);
   
   // Also show the full path for easy copy
   string terminalPath = TerminalInfoString(TERMINAL_DATA_PATH);
   Print("Full path: ", terminalPath, "\\MQL5\\Files\\", filename);
   
   Alert("Data exported successfully!\n", copied, " candles saved to:\n", filename);
}
//+------------------------------------------------------------------+
