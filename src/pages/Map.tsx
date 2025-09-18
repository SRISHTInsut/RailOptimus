// src/components/Map.tsx
import React from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix for default marker icon issue in Leaflet + React
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

const DefaultIcon = L.icon({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

L.Marker.prototype.options.icon = DefaultIcon;

const Map: React.FC = () => {
  return (
    <div className="w-full h-[500px]">
      <MapContainer
        center={[28.6139, 77.2090]} // Default center: New Delhi
        zoom={12}
        scrollWheelZoom={true}
        className="w-full h-full rounded-lg shadow-md"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={[28.6139, 77.2090]}>
          <Popup>New Delhi 🚉</Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};

export default Map;