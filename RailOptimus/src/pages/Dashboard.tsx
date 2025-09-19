import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Train, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Activity,
  TrendingUp,
  Users
} from "lucide-react";

// Mock data
const mockTrains = [
  { id: "T001", type: "Express", scheduled: "09:15", actual: "09:15", delay: 0, status: "on-time" },
  { id: "T002", type: "Local", scheduled: "09:30", actual: "09:35", delay: 5, status: "minor-delay" },
  { id: "T003", type: "Freight", scheduled: "09:45", actual: "10:15", delay: 30, status: "major-delay" },
  { id: "T004", type: "Express", scheduled: "10:00", actual: "10:00", delay: 0, status: "on-time" },
  { id: "T005", type: "Local", scheduled: "10:15", actual: "10:25", delay: 10, status: "minor-delay" },
];

const mockAlerts = [
  { id: 1, type: "congestion", message: "Heavy traffic detected on Block B2-B3", priority: "high" },
  { id: 2, type: "conflict", message: "Potential scheduling conflict at Platform 2", priority: "medium" },
  { id: 3, type: "maintenance", message: "Scheduled maintenance on Track 3 at 14:00", priority: "low" },
];

const mockSuggestions = [
  { 
    id: 1, 
    title: "Reschedule T003 departure", 
    description: "Delay T003 by 15 minutes to optimize platform usage",
    impact: "Reduces overall delay by 8 minutes"
  },
  { 
    id: 2, 
    title: "Route T005 via Platform 1", 
    description: "Alternative platform allocation to reduce congestion",
    impact: "Improves throughput by 12%"
  },
];

export default function Dashboard() {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "on-time":
        return "success";
      case "minor-delay":
        return "warning";
      case "major-delay":
        return "danger";
      default:
        return "secondary";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "on-time":
        return <CheckCircle className="h-4 w-4" />;
      case "minor-delay":
        return <Clock className="h-4 w-4" />;
      case "major-delay":
        return <XCircle className="h-4 w-4" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "danger";
      case "medium":
        return "warning";
      case "low":
        return "secondary";
      default:
        return "secondary";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Control Dashboard</h1>
          <p className="text-muted-foreground">
            Real-time railway section monitoring and control
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-muted-foreground">Current Time</p>
          <p className="text-2xl font-mono font-bold text-foreground">
            {currentTime.toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="border-border shadow-soft">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Trains</p>
                <p className="text-2xl font-bold text-foreground">{mockTrains.length}</p>
              </div>
              <div className="flex items-center justify-center w-12 h-12 bg-primary/10 rounded-lg">
                <Train className="h-6 w-6 text-primary" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border shadow-soft">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">On Time</p>
                <p className="text-2xl font-bold text-success">
                  {mockTrains.filter(t => t.status === "on-time").length}
                </p>
              </div>
              <div className="flex items-center justify-center w-12 h-12 bg-success/10 rounded-lg">
                <CheckCircle className="h-6 w-6 text-success" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border shadow-soft">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Delayed</p>
                <p className="text-2xl font-bold text-warning">
                  {mockTrains.filter(t => t.delay > 0).length}
                </p>
              </div>
              <div className="flex items-center justify-center w-12 h-12 bg-warning/10 rounded-lg">
                <Clock className="h-6 w-6 text-warning" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border shadow-soft">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Alerts</p>
                <p className="text-2xl font-bold text-danger">{mockAlerts.length}</p>
              </div>
              <div className="flex items-center justify-center w-12 h-12 bg-danger/10 rounded-lg">
                <AlertTriangle className="h-6 w-6 text-danger" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Train List */}
        <div className="lg:col-span-2">
          <Card className="border-border shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Train className="h-5 w-5" />
                Live Train Tracking
              </CardTitle>
              <CardDescription>
                Real-time status of trains in your section
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockTrains.map((train) => (
                  <div
                    key={train.id}
                    className="flex items-center justify-between p-4 border border-border rounded-lg bg-card hover:shadow-soft transition-shadow"
                  >
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(train.status)}
                        <span className="font-semibold text-foreground">{train.id}</span>
                      </div>
                      <div>
                        <p className="font-medium text-foreground">{train.type}</p>
                        <p className="text-sm text-muted-foreground">
                          Scheduled: {train.scheduled} | Actual: {train.actual}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant={getStatusColor(train.status) as any}>
                        {train.delay === 0 ? "On Time" : `+${train.delay}m`}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Alerts and Suggestions */}
        <div className="space-y-6">
          {/* Alerts */}
          <Card className="border-border shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Active Alerts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {mockAlerts.map((alert) => (
                <Alert key={alert.id} className="border-border">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="ml-2">
                    <div className="flex items-start justify-between">
                      <span className="text-sm">{alert.message}</span>
                      <Badge 
                        variant={getPriorityColor(alert.priority) as any}
                        className="ml-2 text-xs"
                      >
                        {alert.priority}
                      </Badge>
                    </div>
                  </AlertDescription>
                </Alert>
              ))}
            </CardContent>
          </Card>

          {/* AI Suggestions */}
          <Card className="border-border shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                AI Suggestions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {mockSuggestions.map((suggestion) => (
                <div key={suggestion.id} className="p-4 border border-border rounded-lg bg-muted/30">
                  <h4 className="font-medium text-foreground mb-2">{suggestion.title}</h4>
                  <p className="text-sm text-muted-foreground mb-2">{suggestion.description}</p>
                  <p className="text-xs text-success mb-3">{suggestion.impact}</p>
                  <div className="flex gap-2">
                    <Button size="sm" variant="success">Approve</Button>
                    <Button size="sm" variant="outline">Modify</Button>
                    <Button size="sm" variant="ghost">Reject</Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}