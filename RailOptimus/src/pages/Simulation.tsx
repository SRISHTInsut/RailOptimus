import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { 
  Play, 
  RotateCcw, 
  Save, 
  ChevronDown, 
  ChevronUp, 
  BarChart3, 
  Clock, 
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Download,
  FileText
} from "lucide-react";
import { exportToCSV, exportToPDF } from "@/lib/exportUtils";

interface SimulationHistory {
  id: string;
  name: string;
  date: string;
  totalDelay: number;
  conflicts: number;
  affectedTrains: number;
  status: "completed" | "running" | "failed";
}

interface SimulationResults {
  throughput: { before: number; after: number };
  totalDelay: { before: number; after: number };
  conflicts: { before: number; after: number };
  affectedTrains: {
    minorDelay: number;
    majorDelay: number;
    routeChange: number;
  };
}

const mockTrains = [
  "TR001 - Express Mumbai-Delhi",
  "TR002 - Local Suburban",
  "TR003 - Freight Goods",
  "TR004 - Express Chennai-Kolkata",
  "TR005 - Mail Express"
];

const mockBlocks = [
  "Block A1 - Main Line",
  "Block A2 - Junction",
  "Block B1 - Loop Line",
  "Block B2 - Platform",
  "Block C1 - Yard"
];

const mockHistory: SimulationHistory[] = [
  { id: "1", name: "Morning Rush Delay", date: "2024-01-15", totalDelay: 45, conflicts: 2, affectedTrains: 8, status: "completed" },
  { id: "2", name: "Block A1 Closure", date: "2024-01-14", totalDelay: 120, conflicts: 5, affectedTrains: 12, status: "completed" },
  { id: "3", name: "Express Priority Test", date: "2024-01-13", totalDelay: 20, conflicts: 1, affectedTrains: 3, status: "completed" },
];

export default function Simulation() {
  const [scenarioType, setScenarioType] = useState("");
  const [selectedTrain, setSelectedTrain] = useState("");
  const [selectedBlock, setSelectedBlock] = useState("");
  const [delayMinutes, setDelayMinutes] = useState("");
  const [isAdditionalOpen, setIsAdditionalOpen] = useState(false);
  const [simulationResults, setSimulationResults] = useState<SimulationResults | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [exportFormat, setExportFormat] = useState("csv");

  // Additional constraints state
  const [blockCapacity, setBlockCapacity] = useState("1");
  const [minHeadway, setMinHeadway] = useState("120");
  const [minOccupyTime, setMinOccupyTime] = useState("60");
  const [platformCapacity, setplatformCapacity] = useState("2");
  const [dwellTime, setDwellTime] = useState("90");
  const [maxDelay, setMaxDelay] = useState("180");

  const runSimulation = async () => {
    setIsRunning(true);
    
    // Open RailOptiSim in a popup window
    const popupWindow = window.open(
      'http://localhost:8050',
      'RailOptiSim',
      'width=1400,height=900,scrollbars=yes,resizable=yes,toolbar=no,menubar=no,location=no,status=no'
    );
    
    // Focus the popup window
    if (popupWindow) {
      popupWindow.focus();
    }
    
    // Simulate API call for mock results (keeping existing functionality)
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Calculate results based on scenario builder inputs
    const delayValue = parseInt(delayMinutes) || 0;
    const blockCap = parseInt(blockCapacity) || 1;
    const headway = parseInt(minHeadway) || 120;
    const platformCap = parseInt(platformCapacity) || 2;
    
    // Dynamic calculation based on inputs
    const baseThroughput = 85;
    const throughputImpact = Math.max(0, baseThroughput - (delayValue * 2) - (blockCap > 1 ? 10 : 0));
    
    const baseDelay = 25;
    const calculatedDelay = baseDelay + delayValue + (headway > 120 ? 5 : 0);
    
    const baseConflicts = 1;
    const calculatedConflicts = baseConflicts + (delayValue > 30 ? 2 : 0) + (platformCap > 2 ? 1 : 0);
    
    // Mock results based on actual inputs
    setSimulationResults({
      throughput: { before: baseThroughput, after: throughputImpact },
      totalDelay: { before: baseDelay, after: calculatedDelay },
      conflicts: { before: baseConflicts, after: calculatedConflicts },
      affectedTrains: {
        minorDelay: Math.max(1, Math.floor(delayValue / 10)),
        majorDelay: delayValue > 30 ? Math.floor(delayValue / 20) : 0,
        routeChange: scenarioType === "block" || scenarioType === "accident" ? 2 : 1
      }
    });
    setIsRunning(false);
  };

  const resetScenario = () => {
    setScenarioType("");
    setSelectedTrain("");
    setSelectedBlock("");
    setDelayMinutes("");
    setBlockCapacity("1");
    setMinHeadway("120");
    setMinOccupyTime("60");
    setplatformCapacity("2");
    setDwellTime("90");
    setMaxDelay("180");
    setSimulationResults(null);
  };

  const saveScenario = () => {
    const scenarioData = {
      scenarioType,
      selectedTrain,
      selectedBlock,
      delayMinutes,
      blockCapacity,
      minHeadway,
      minOccupyTime,
      platformCapacity,
      dwellTime,
      maxDelay,
      timestamp: new Date().toISOString()
    };
    
    // Save to localStorage for persistence
    localStorage.setItem('savedScenario', JSON.stringify(scenarioData));
    
    // You could also show a toast notification here
    console.log('Scenario saved:', scenarioData);
  };

  const downloadSimulationResults = async () => {
    if (!simulationResults) return;

    const exportData = {
      selectedDate: new Date().toISOString().split('T')[0],
      reportType: "simulation",
      summary: {
        date: new Date().toISOString().split('T')[0],
        handledDelays: simulationResults.affectedTrains.minorDelay + simulationResults.affectedTrains.majorDelay,
        resolvedConflicts: simulationResults.conflicts.after,
        emergencies: simulationResults.conflicts.after - simulationResults.conflicts.before,
        approvedSuggestions: simulationResults.affectedTrains.routeChange,
        rejectedSuggestions: 0,
        totalSuggestions: simulationResults.affectedTrains.routeChange
      },
      suggestions: [
        {
          id: "1",
          time: new Date().toLocaleTimeString(),
          type: "simulation",
          description: `Simulation: ${scenarioType} scenario with ${delayMinutes} minutes delay`,
          status: "completed",
          impact: simulationResults.totalDelay.after > simulationResults.totalDelay.before ? "high" : "medium"
        }
      ],
      simulationParameters: {
        scenarioType,
        selectedTrain,
        selectedBlock,
        delayMinutes,
        blockCapacity,
        minHeadway,
        minOccupyTime,
        platformCapacity,
        dwellTime,
        maxDelay
      },
      results: simulationResults
    };

    if (exportFormat === "csv") {
      exportToCSV(exportData);
    } else if (exportFormat === "pdf") {
      await exportToPDF(exportData, "simulation-results-card");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Traffic Simulation</h1>
        <p className="text-muted-foreground">Create and test railway traffic scenarios</p>
      </div>

      {/* Main Simulation Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scenario Builder */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Scenario Builder
            </CardTitle>
            <CardDescription>Configure simulation parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Major Adjustments */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Major Adjustments</h3>
              
              <div className="space-y-2">
                <Label htmlFor="scenario-type">Scenario Type</Label>
                <Select value={scenarioType} onValueChange={setScenarioType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose scenario type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="delay">Delay a train</SelectItem>
                    <SelectItem value="block">Block a section</SelectItem>
                    <SelectItem value="accident">Trigger an accident</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {scenarioType === "delay" && (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="train-select">Train Selection</Label>
                    <Select value={selectedTrain} onValueChange={setSelectedTrain}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select train" />
                      </SelectTrigger>
                      <SelectContent>
                        {mockTrains.map((train, index) => (
                          <SelectItem key={index} value={train}>{train}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="delay">Delay (minutes)</Label>
                    <Input
                      id="delay"
                      type="number"
                      value={delayMinutes}
                      onChange={(e) => setDelayMinutes(e.target.value)}
                      placeholder="Enter delay in minutes"
                    />
                  </div>
                </>
              )}

              {(scenarioType === "block" || scenarioType === "accident") && (
                <div className="space-y-2">
                  <Label htmlFor="block-select">Block Section</Label>
                  <Select value={selectedBlock} onValueChange={setSelectedBlock}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select block" />
                    </SelectTrigger>
                    <SelectContent>
                      {mockBlocks.map((block, index) => (
                        <SelectItem key={index} value={block}>{block}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="start-time">Start Time</Label>
                  <Input id="start-time" type="time" defaultValue="09:00" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="end-time">End Time</Label>
                  <Input id="end-time" type="time" defaultValue="11:00" />
                </div>
              </div>
            </div>

            <Separator />

            {/* Additional Changes */}
            <Collapsible open={isAdditionalOpen} onOpenChange={setIsAdditionalOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between p-0 h-auto">
                  <span className="text-primary underline">Additional Changes</span>
                  {isAdditionalOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-4 mt-4">
                <Tabs defaultValue="safety" className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="safety">Safety Constraints</TabsTrigger>
                    <TabsTrigger value="operational">Operational</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="safety" className="space-y-4">
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-destructive" />
                        <span className="text-sm font-medium text-destructive">Must-Have Safety Constraints</span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="block-capacity">Block Capacity</Label>
                          <Input
                            id="block-capacity"
                            type="number"
                            value={blockCapacity}
                            onChange={(e) => setBlockCapacity(e.target.value)}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="min-headway">Min Headway (seconds)</Label>
                          <Input
                            id="min-headway"
                            type="number"
                            value={minHeadway}
                            onChange={(e) => setMinHeadway(e.target.value)}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="min-occupy">Min Block Occupation (seconds)</Label>
                          <Input
                            id="min-occupy"
                            type="number"
                            value={minOccupyTime}
                            onChange={(e) => setMinOccupyTime(e.target.value)}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="platform-capacity">Platform Capacity</Label>
                          <Input
                            id="platform-capacity"
                            type="number"
                            value={platformCapacity}
                            onChange={(e) => setplatformCapacity(e.target.value)}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="dwell-time">Dwell Time (seconds)</Label>
                          <Input
                            id="dwell-time"
                            type="number"
                            value={dwellTime}
                            onChange={(e) => setDwellTime(e.target.value)}
                          />
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="operational" className="space-y-4">
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-warning" />
                        <span className="text-sm font-medium text-warning">Essential Operational Constraints</span>
                      </div>
                      
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="max-delay">Maximum Allowed Delay (minutes)</Label>
                          <Input
                            id="max-delay"
                            type="number"
                            value={maxDelay}
                            onChange={(e) => setMaxDelay(e.target.value)}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Priority Ordering</Label>
                          <Select defaultValue="express">
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="express">Express First</SelectItem>
                              <SelectItem value="scheduled">By Schedule</SelectItem>
                              <SelectItem value="passenger">Passenger Priority</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CollapsibleContent>
            </Collapsible>

            <Separator />

            {/* Action Buttons */}
            <div className="flex gap-2">
              <Button 
                onClick={runSimulation} 
                disabled={!scenarioType || isRunning}
                className="flex-1"
              >
                <Play className="h-4 w-4 mr-2" />
                {isRunning ? "Running..." : "Run Simulation"}
              </Button>
              <Button variant="outline" onClick={resetScenario}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Reset
              </Button>
               <Button variant="outline" onClick={saveScenario}>
                 <Save className="h-4 w-4 mr-2" />
                 Save
               </Button>
            </div>
          </CardContent>
        </Card>

        {/* Simulation Results */}
        <Card id="simulation-results-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Simulation Results
            </CardTitle>
            <CardDescription>Before/After scenario comparison</CardDescription>
          </CardHeader>
          <CardContent>
            {simulationResults ? (
              <div className="space-y-6">
                {/* KPIs */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Throughput</Label>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">
                        {simulationResults.throughput.before}% → {simulationResults.throughput.after}%
                      </span>
                      {simulationResults.throughput.after < simulationResults.throughput.before ? (
                        <TrendingDown className="h-4 w-4 text-destructive" />
                      ) : (
                        <TrendingUp className="h-4 w-4 text-success" />
                      )}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Total Delay (min)</Label>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">
                        {simulationResults.totalDelay.before} → {simulationResults.totalDelay.after}
                      </span>
                      {simulationResults.totalDelay.after > simulationResults.totalDelay.before ? (
                        <TrendingUp className="h-4 w-4 text-destructive" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-success" />
                      )}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">Conflicts</Label>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">
                        {simulationResults.conflicts.before} → {simulationResults.conflicts.after}
                      </span>
                      {simulationResults.conflicts.after > simulationResults.conflicts.before ? (
                        <TrendingUp className="h-4 w-4 text-destructive" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-success" />
                      )}
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Affected Trains */}
                <div className="space-y-4">
                  <Label className="text-sm font-medium">Affected Trains</Label>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-warning">{simulationResults.affectedTrains.minorDelay}</div>
                      <div className="text-sm text-muted-foreground">Minor Delay</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-destructive">{simulationResults.affectedTrains.majorDelay}</div>
                      <div className="text-sm text-muted-foreground">Major Delay</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">{simulationResults.affectedTrains.routeChange}</div>
                      <div className="text-sm text-muted-foreground">Route Change</div>
                    </div>
                  </div>
                </div>

                 <div className="flex gap-2">
                   <Button className="flex-1">
                     <BarChart3 className="h-4 w-4 mr-2" />
                     Detailed Analysis
                   </Button>
                   <div className="flex-1 flex gap-1">
                     <Select value={exportFormat} onValueChange={setExportFormat}>
                       <SelectTrigger className="w-20">
                         <SelectValue />
                       </SelectTrigger>
                       <SelectContent>
                         <SelectItem value="csv">CSV</SelectItem>
                         <SelectItem value="pdf">PDF</SelectItem>
                       </SelectContent>
                     </Select>
                     <Button 
                       onClick={downloadSimulationResults}
                       variant="outline"
                       className="flex-1"
                     >
                       <FileText className="h-4 w-4 mr-2" />
                       Download
                     </Button>
                   </div>
                 </div>
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Run a simulation to see results</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Simulation History */}
      <Card>
        <CardHeader>
          <CardTitle>Simulation History</CardTitle>
          <CardDescription>Last 10 simulation scenarios</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {mockHistory.map((simulation) => (
              <div key={simulation.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="space-y-1">
                  <div className="font-medium">{simulation.name}</div>
                  <div className="text-sm text-muted-foreground">{simulation.date}</div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-sm">
                    <span className="font-medium">{simulation.totalDelay}min</span> delay
                  </div>
                  <div className="text-sm">
                    <span className="font-medium">{simulation.conflicts}</span> conflicts
                  </div>
                  <div className="text-sm">
                    <span className="font-medium">{simulation.affectedTrains}</span> affected
                  </div>
                  <Badge variant={simulation.status === "completed" ? "success" : "secondary"}>
                    {simulation.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}