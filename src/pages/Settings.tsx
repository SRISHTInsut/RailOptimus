import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { useAuthStore } from "@/stores/authStore";
import { 
  User, 
  Settings as SettingsIcon, 
  Bell, 
  Eye, 
  Shield, 
  Save,
  UserCircle,
  MapPin,
  Mail,
  Phone
} from "lucide-react";

export default function Settings() {
  const { user, updateUser } = useAuthStore();
  const { toast } = useToast();
  
  // User Profile State
  const [name, setName] = useState(user?.name || "");
  const [section, setSection] = useState(user?.section || "");
  const [email, setEmail] = useState(user?.email || "controller@railway.gov");
  const [phone, setPhone] = useState(user?.phone || "+91-9876543210");
  const [employeeId, setEmployeeId] = useState(user?.employeeId || "RC001");
  
  // Preferences State
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(true);
  const [soundAlerts, setSoundAlerts] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState("30");
  const [theme, setTheme] = useState("system");
  const [language, setLanguage] = useState("en");
  
  // Visualization Preferences
  const [showTrainNumbers, setShowTrainNumbers] = useState(true);
  const [showDelayColors, setShowDelayColors] = useState(true);
  const [showBlockStatus, setShowBlockStatus] = useState(true);
  const [mapZoomLevel, setMapZoomLevel] = useState("medium");
  const [animateTrains, setAnimateTrains] = useState(false);

  const handleSaveProfile = () => {
    const updatedUser = {
      ...user,
      name,
      section,
      email,
      phone,
      employeeId
    };
    
    updateUser(updatedUser);
    toast({
      title: "Profile Updated",
      description: "Your profile information has been saved successfully.",
    });
  };

  const handleSavePreferences = () => {
    // In a real app, this would save to backend/localStorage
    toast({
      title: "Preferences Saved",
      description: "Your preferences have been updated successfully.",
    });
  };

  const handleResetSettings = () => {
    // Reset to defaults
    setEmailNotifications(true);
    setPushNotifications(true);
    setSoundAlerts(false);
    setAutoRefresh(true);
    setRefreshInterval("30");
    setTheme("system");
    setLanguage("en");
    setShowTrainNumbers(true);
    setShowDelayColors(true);
    setShowBlockStatus(true);
    setMapZoomLevel("medium");
    setAnimateTrains(false);
    
    toast({
      title: "Settings Reset",
      description: "All settings have been reset to default values.",
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Settings</h1>
        <p className="text-muted-foreground">Manage your account and application preferences</p>
      </div>

      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="profile">Profile</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        {/* Profile Settings */}
        <TabsContent value="profile">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserCircle className="h-5 w-5" />
                Profile Information
              </CardTitle>
              <CardDescription>
                Update your personal information and section details
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="pl-10"
                      placeholder="Enter your full name"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="employee-id">Employee ID</Label>
                  <Input
                    id="employee-id"
                    value={employeeId}
                    onChange={(e) => setEmployeeId(e.target.value)}
                    placeholder="Employee ID"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="section">Section</Label>
                  <div className="relative">
                    <MapPin className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="section"
                      value={section}
                      onChange={(e) => setSection(e.target.value)}
                      className="pl-10"
                      placeholder="Section assignment"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10"
                      placeholder="Email address"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number</Label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="phone"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      className="pl-10"
                      placeholder="Phone number"
                    />
                  </div>
                </div>
              </div>

              <Separator />

              <div className="flex justify-end">
                <Button onClick={handleSaveProfile}>
                  <Save className="h-4 w-4 mr-2" />
                  Save Profile
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notification Settings */}
        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Notification Preferences
              </CardTitle>
              <CardDescription>
                Configure how you receive alerts and updates
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="email-notifications">Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive important alerts via email
                    </p>
                  </div>
                  <Switch
                    id="email-notifications"
                    checked={emailNotifications}
                    onCheckedChange={setEmailNotifications}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="push-notifications">Push Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Get real-time notifications in your browser
                    </p>
                  </div>
                  <Switch
                    id="push-notifications"
                    checked={pushNotifications}
                    onCheckedChange={setPushNotifications}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="sound-alerts">Sound Alerts</Label>
                    <p className="text-sm text-muted-foreground">
                      Play sound for critical alerts
                    </p>
                  </div>
                  <Switch
                    id="sound-alerts"
                    checked={soundAlerts}
                    onCheckedChange={setSoundAlerts}
                  />
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <h3 className="text-lg font-medium">Auto-Refresh Settings</h3>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="auto-refresh">Auto-Refresh Dashboard</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically update dashboard data
                    </p>
                  </div>
                  <Switch
                    id="auto-refresh"
                    checked={autoRefresh}
                    onCheckedChange={setAutoRefresh}
                  />
                </div>

                {autoRefresh && (
                  <div className="space-y-2">
                    <Label htmlFor="refresh-interval">Refresh Interval</Label>
                    <Select value={refreshInterval} onValueChange={setRefreshInterval}>
                      <SelectTrigger className="w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="10">10 seconds</SelectItem>
                        <SelectItem value="30">30 seconds</SelectItem>
                        <SelectItem value="60">1 minute</SelectItem>
                        <SelectItem value="300">5 minutes</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>

              <Separator />

              <div className="flex justify-end">
                <Button onClick={handleSavePreferences}>
                  <Save className="h-4 w-4 mr-2" />
                  Save Preferences
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Visualization Settings */}
        <TabsContent value="visualization">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Visualization Preferences
              </CardTitle>
              <CardDescription>
                Customize how information is displayed
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Dashboard Display</h3>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="show-train-numbers">Show Train Numbers</Label>
                    <p className="text-sm text-muted-foreground">
                      Display train numbers in the tracking view
                    </p>
                  </div>
                  <Switch
                    id="show-train-numbers"
                    checked={showTrainNumbers}
                    onCheckedChange={setShowTrainNumbers}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="show-delay-colors">Color-coded Delays</Label>
                    <p className="text-sm text-muted-foreground">
                      Use colors to indicate delay severity
                    </p>
                  </div>
                  <Switch
                    id="show-delay-colors"
                    checked={showDelayColors}
                    onCheckedChange={setShowDelayColors}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="show-block-status">Block Status Indicators</Label>
                    <p className="text-sm text-muted-foreground">
                      Show real-time block occupancy status
                    </p>
                  </div>
                  <Switch
                    id="show-block-status"
                    checked={showBlockStatus}
                    onCheckedChange={setShowBlockStatus}
                  />
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <h3 className="text-lg font-medium">Map Settings</h3>
                
                <div className="space-y-2">
                  <Label htmlFor="map-zoom">Default Map Zoom Level</Label>
                  <Select value={mapZoomLevel} onValueChange={setMapZoomLevel}>
                    <SelectTrigger className="w-48">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low (Overview)</SelectItem>
                      <SelectItem value="medium">Medium (Balanced)</SelectItem>
                      <SelectItem value="high">High (Detailed)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="animate-trains">Animate Train Movement</Label>
                    <p className="text-sm text-muted-foreground">
                      Show smooth animations for train movement (may impact performance)
                    </p>
                  </div>
                  <Switch
                    id="animate-trains"
                    checked={animateTrains}
                    onCheckedChange={setAnimateTrains}
                  />
                </div>
              </div>

              <Separator />

              <div className="flex justify-end">
                <Button onClick={handleSavePreferences}>
                  <Save className="h-4 w-4 mr-2" />
                  Save Preferences
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* System Settings */}
        <TabsContent value="system">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <SettingsIcon className="h-5 w-5" />
                System Settings
              </CardTitle>
              <CardDescription>
                Application and security settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="theme">Theme</Label>
                  <Select value={theme} onValueChange={setTheme}>
                    <SelectTrigger className="w-48">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="language">Language</Label>
                  <Select value={language} onValueChange={setLanguage}>
                    <SelectTrigger className="w-48">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="hi">Hindi</SelectItem>
                      <SelectItem value="te">Telugu</SelectItem>
                      <SelectItem value="ta">Tamil</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  <h3 className="text-lg font-medium">Security</h3>
                </div>
                
                <div className="space-y-4">
                  <Button variant="outline" className="w-full justify-start">
                    Change Password
                  </Button>
                  
                  <Button variant="outline" className="w-full justify-start">
                    Security Logs
                  </Button>
                  
                  <Button variant="outline" className="w-full justify-start">
                    Active Sessions
                  </Button>
                </div>
              </div>

              <Separator />

              <div className="flex justify-between">
                <Button variant="outline" onClick={handleResetSettings}>
                  Reset to Defaults
                </Button>
                <Button onClick={handleSavePreferences}>
                  <Save className="h-4 w-4 mr-2" />
                  Save All Settings
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}