import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import Papa from 'papaparse';

export interface ReportExportData {
  selectedDate: string;
  reportType: string;
  summary: {
    date: string;
    handledDelays: number;
    resolvedConflicts: number;
    emergencies: number;
    approvedSuggestions: number;
    rejectedSuggestions: number;
    totalSuggestions: number;
  };
  suggestions: Array<{
    id: string;
    time: string;
    type: string;
    description: string;
    status: string;
    impact: string;
  }>;
  simulationParameters?: {
    scenarioType: string;
    selectedTrain: string;
    selectedBlock: string;
    delayMinutes: string;
    blockCapacity: string;
    minHeadway: string;
    minOccupyTime: string;
    platformCapacity: string;
    dwellTime: string;
    maxDelay: string;
  };
  results?: {
    throughput: { before: number; after: number };
    totalDelay: { before: number; after: number };
    conflicts: { before: number; after: number };
    affectedTrains: {
      minorDelay: number;
      majorDelay: number;
      routeChange: number;
    };
  };
}

export const exportToCSV = (data: ReportExportData) => {
  const csvData = [
    ['RailOptimus Report', ''],
    ['Generated at', new Date().toLocaleString()],
    ['Report Date', data.selectedDate],
    ['Report Type', data.reportType],
    ['', ''],
  ];

  // Add simulation parameters if available
  if (data.simulationParameters) {
    csvData.push(
      ['Simulation Parameters', ''],
      ['Scenario Type', data.simulationParameters.scenarioType],
      ['Selected Train', data.simulationParameters.selectedTrain],
      ['Selected Block', data.simulationParameters.selectedBlock],
      ['Delay Minutes', data.simulationParameters.delayMinutes],
      ['Block Capacity', data.simulationParameters.blockCapacity],
      ['Min Headway (seconds)', data.simulationParameters.minHeadway],
      ['Min Occupy Time (seconds)', data.simulationParameters.minOccupyTime],
      ['Platform Capacity', data.simulationParameters.platformCapacity],
      ['Dwell Time (seconds)', data.simulationParameters.dwellTime],
      ['Max Delay (minutes)', data.simulationParameters.maxDelay],
      ['', '']
    );
  }

  // Add simulation results if available
  if (data.results) {
    csvData.push(
      ['Simulation Results', ''],
      ['Throughput Before (%)', data.results.throughput.before.toString()],
      ['Throughput After (%)', data.results.throughput.after.toString()],
      ['Total Delay Before (min)', data.results.totalDelay.before.toString()],
      ['Total Delay After (min)', data.results.totalDelay.after.toString()],
      ['Conflicts Before', data.results.conflicts.before.toString()],
      ['Conflicts After', data.results.conflicts.after.toString()],
      ['Minor Delays', data.results.affectedTrains.minorDelay.toString()],
      ['Major Delays', data.results.affectedTrains.majorDelay.toString()],
      ['Route Changes', data.results.affectedTrains.routeChange.toString()],
      ['', '']
    );
  }

  csvData.push(
    ['Daily Summary', ''],
    ['Date', data.summary.date],
    ['Handled Delays', data.summary.handledDelays.toString()],
    ['Resolved Conflicts', data.summary.resolvedConflicts.toString()],
    ['Emergencies', data.summary.emergencies.toString()],
    ['Approved Suggestions', data.summary.approvedSuggestions.toString()],
    ['Rejected Suggestions', data.summary.rejectedSuggestions.toString()],
    ['Total Suggestions', data.summary.totalSuggestions.toString()],
    ['Approval Rate', ((data.summary.approvedSuggestions / data.summary.totalSuggestions) * 100).toFixed(1) + '%'],
    ['', ''],
    ['AI Suggestions Detail', ''],
    ['Time', 'Type', 'Description', 'Impact', 'Status']
  );

  // Add suggestion details
  data.suggestions.forEach(suggestion => {
    csvData.push([
      suggestion.time,
      suggestion.type,
      suggestion.description,
      suggestion.impact,
      suggestion.status
    ]);
  });

  const csv = Papa.unparse(csvData);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', `railway_report_${data.selectedDate}.csv`);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const exportToPDF = async (data: ReportExportData, elementId?: string) => {
  const pdf = new jsPDF();
  
  // Add title
  pdf.setFontSize(20);
  pdf.text('RailOptimus Simulation Report', 20, 30);
  
  // Add timestamp and report info
  pdf.setFontSize(12);
  pdf.text(`Generated: ${new Date().toLocaleString()}`, 20, 40);
  pdf.text(`Report Date: ${data.selectedDate}`, 20, 50);
  pdf.text(`Report Type: ${data.reportType}`, 20, 60);
  
  let yPosition = 80;
  const lineHeight = 7;
  
  // Add simulation parameters if available
  if (data.simulationParameters) {
    pdf.setFontSize(16);
    pdf.text('Simulation Parameters', 20, yPosition);
    yPosition += lineHeight + 5;
    
    pdf.setFontSize(10);
    pdf.text(`Scenario Type: ${data.simulationParameters.scenarioType}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Selected Train: ${data.simulationParameters.selectedTrain}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Selected Block: ${data.simulationParameters.selectedBlock}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Delay Minutes: ${data.simulationParameters.delayMinutes}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Block Capacity: ${data.simulationParameters.blockCapacity}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Min Headway: ${data.simulationParameters.minHeadway} seconds`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Min Occupy Time: ${data.simulationParameters.minOccupyTime} seconds`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Platform Capacity: ${data.simulationParameters.platformCapacity}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Dwell Time: ${data.simulationParameters.dwellTime} seconds`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Max Delay: ${data.simulationParameters.maxDelay} minutes`, 20, yPosition);
    yPosition += lineHeight + 10;
  }
  
  // Add simulation results if available
  if (data.results) {
    pdf.setFontSize(16);
    pdf.text('Simulation Results', 20, yPosition);
    yPosition += lineHeight + 5;
    
    pdf.setFontSize(10);
    pdf.text(`Throughput: ${data.results.throughput.before}% → ${data.results.throughput.after}%`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Total Delay: ${data.results.totalDelay.before} → ${data.results.totalDelay.after} minutes`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`Conflicts: ${data.results.conflicts.before} → ${data.results.conflicts.after}`, 20, yPosition);
    yPosition += lineHeight;
    
    pdf.text('Affected Trains:', 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`  Minor Delays: ${data.results.affectedTrains.minorDelay}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`  Major Delays: ${data.results.affectedTrains.majorDelay}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`  Route Changes: ${data.results.affectedTrains.routeChange}`, 20, yPosition);
    yPosition += lineHeight + 10;
  }
  
  // Add daily summary
  pdf.setFontSize(16);
  pdf.text('Daily Summary', 20, yPosition);
  yPosition += lineHeight + 5;
  
  pdf.setFontSize(10);
  pdf.text(`Date: ${data.summary.date}`, 20, yPosition);
  yPosition += lineHeight;
  pdf.text(`Handled Delays: ${data.summary.handledDelays}`, 20, yPosition);
  yPosition += lineHeight;
  pdf.text(`Resolved Conflicts: ${data.summary.resolvedConflicts}`, 20, yPosition);
  yPosition += lineHeight;
  pdf.text(`Emergencies: ${data.summary.emergencies}`, 20, yPosition);
  yPosition += lineHeight;
  pdf.text(`Approved Suggestions: ${data.summary.approvedSuggestions}`, 20, yPosition);
  yPosition += lineHeight;
  pdf.text(`Rejected Suggestions: ${data.summary.rejectedSuggestions}`, 20, yPosition);
  yPosition += lineHeight;
  pdf.text(`Total Suggestions: ${data.summary.totalSuggestions}`, 20, yPosition);
  yPosition += lineHeight;
  pdf.text(`Approval Rate: ${((data.summary.approvedSuggestions / data.summary.totalSuggestions) * 100).toFixed(1)}%`, 20, yPosition);
  
  // Add suggestions detail
  yPosition += lineHeight + 10;
  pdf.setFontSize(16);
  pdf.text('AI Suggestions Detail', 20, yPosition);
  yPosition += lineHeight + 5;
  
  pdf.setFontSize(10);
  data.suggestions.forEach((suggestion, index) => {
    if (yPosition > 250) {
      pdf.addPage();
      yPosition = 30;
    }
    
    pdf.text(`${index + 1}. ${suggestion.time} - ${suggestion.type}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`   Description: ${suggestion.description}`, 20, yPosition);
    yPosition += lineHeight;
    pdf.text(`   Impact: ${suggestion.impact} | Status: ${suggestion.status}`, 20, yPosition);
    yPosition += lineHeight + 3;
  });
  
  // If elementId is provided, try to capture the element as an image
  if (elementId) {
    try {
      const element = document.getElementById(elementId);
      if (element) {
        const canvas = await html2canvas(element, {
          scale: 2,
          useCORS: true,
          allowTaint: true
        });
        
        const imgData = canvas.toDataURL('image/png');
        
        // Add new page for the chart
        pdf.addPage();
        pdf.setFontSize(16);
        pdf.text('Simulation Visualization', 20, 30);
        
        // Calculate image dimensions to fit the page
        const imgWidth = 170;
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        
        pdf.addImage(imgData, 'PNG', 20, 40, imgWidth, imgHeight);
      }
    } catch (error) {
      console.warn('Could not capture element as image:', error);
    }
  }
  
  // Save the PDF
  pdf.save(`simulation_report_${data.selectedDate}.pdf`);
};
