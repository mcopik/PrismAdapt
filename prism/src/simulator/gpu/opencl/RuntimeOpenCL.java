//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.opencl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Random;

import parser.State;
import parser.ast.Expression;
import parser.ast.LabelList;
import parser.ast.ModulesFile;
import parser.ast.PropertiesFile;
import prism.Pair;
import prism.Preconditions;
import prism.PrismException;
import prism.PrismLog;
import prism.PrismSettings;
import simulator.gpu.RuntimeDeviceInterface;
import simulator.gpu.RuntimeFrameworkInterface;
import simulator.gpu.automaton.AbstractAutomaton;
import simulator.gpu.automaton.command.AdaptCommand;
import simulator.gpu.automaton.command.CommandInterface;
import simulator.gpu.opencl.kernel.PRNGRandom123;
import simulator.method.CIMethod;
import simulator.method.CIwidth;
import simulator.method.SimulationMethod;
import simulator.sampler.Sampler;

import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLException;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;
import com.nativelibs4java.opencl.JavaCL;

public class RuntimeOpenCL implements RuntimeFrameworkInterface
{

	private final CLPlatform[] platforms;
	private final CLDeviceWrapper[] devices;
	private CLPlatform currentPlatform = null;
	private List<CLDeviceWrapper> currentDevices = new ArrayList<>();
	List<RuntimeContext> currentContexts = null;
	private long maxPathLength = 0;
	private State initialState = null;
	private PrismLog mainLog = null;
	private PrismSettings prismSettings = null;
	private int minPathFound = 0;
	private int maxPathFound = 0;
	private float avgPathFound = 0;
	private parser.ast.ModulesFile mf;
	private parser.ast.PropertiesFile pf;
	private Expression expr;

	/**
	 * Constructor. Throws an exception when OpenCL initialization failed.
	 * @throws PrismException
	 */
	public RuntimeOpenCL() throws PrismException
	{
		try {
			platforms = JavaCL.listPlatforms();
			List<CLDeviceWrapper> devs = new ArrayList<>();
			for (CLPlatform platform : platforms) {
				CLDevice[] dev = platform.listAllDevices(true);
				for (CLDevice device : dev) {
					devs.add(new CLDeviceWrapper(device));
				}
			}
			devices = devs.toArray(new CLDeviceWrapper[devs.size()]);
		} catch (CLException exc) {
			throw new PrismException("An error has occured!\n" + exc.getMessage());
		} catch (Exception exc) {
			throw new PrismException("An error has occured!\n" + exc.getMessage());
		} catch (Error err) {
			if (err.getCause() instanceof CLException) {
				CLException exc = (CLException) err.getCause();
				// CL_PLATFORM_NOT_FOUND_KHR
				if (exc.getCode() == -1001) {
					throw new PrismException("None OpenCL platform has not been found!");
				} else {
					throw new PrismException("An error has occured!\n" + exc.getMessage());
				}
			} else {
				throw new PrismException("An error has occured!\n" + err.getMessage());
			}
		}
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getFrameworkName()
	 */
	@Override
	public String getFrameworkName()
	{
		return "OpenCL";
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getPlatformInfo()
	 */
	@Override
	public String getPlatformInfo(int platformNumber)
	{
		Preconditions.checkIndex(platformNumber, platforms.length, String.format("%d is not valid platform number", platformNumber));
		return currentPlatform.getName() + " " + currentPlatform.getVendor();
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getPlatformNames()
	 */
	public String[] getPlatformNames()
	{
		String[] names = new String[platforms.length];
		for (int i = 0; i < platforms.length; ++i) {
			names[i] = platforms[i].getName();
		}
		return names;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getDevices()
	 */
	@Override
	public RuntimeDeviceInterface[] getDevices()
	{
		return devices;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getDevicesNames()
	 */
	@Override
	public String[] getDevicesNames()
	{
		String[] result = new String[devices.length];
		for (int i = 0; i < devices.length; ++i) {
			result[i] = devices[i].getName();
		}
		return result;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getMaxFlopsDevice()
	 */
	@Override
	public RuntimeDeviceInterface getMaxFlopsDevice()
	{
		return new CLDeviceWrapper(JavaCL.getBestDevice());
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getMaxFlopsDevice()
	 */
	@Override
	public RuntimeDeviceInterface getMaxFlopsDevice(DeviceType type)
	{
		if (type == DeviceType.CPU) {
			return new CLDeviceWrapper(JavaCL.getBestDevice(DeviceFeature.CPU));
		} else {
			return new CLDeviceWrapper(JavaCL.getBestDevice(DeviceFeature.GPU));
		}
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#selectDevice()
	 */
	@Override
	public void selectDevice(RuntimeDeviceInterface device)
	{
		Preconditions.checkCondition(device instanceof CLDeviceWrapper, "RuntimeOpenCL can't select non-opencl device");
		currentDevices.add((CLDeviceWrapper) device);
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#getPlatformNumber()
	 */
	@Override
	public int getPlatformNumber()
	{
		return platforms.length;
	}

	public void setMF(ModulesFile mf)
	{
		this.mf = mf;
	}

	public void setPF(PropertiesFile pf)
	{
		this.pf = pf;
	}

	public void setExprs(Expression expr)
	{
		this.expr = expr;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.RuntimeFrameworkInterface#simulateProperty()
	 */
	@Override
	public int simulateProperty(AbstractAutomaton model, List<Sampler> properties) throws PrismException
	{
		Preconditions.checkNotNull(mainLog, "");
		Preconditions.checkCondition(maxPathLength > 0, "");
		RuntimeConfig config = new RuntimeConfig();
		if (initialState != null) {
			config.initialState = initialState;
		}
		config.maxPathLength = maxPathLength;
		if (prismSettings == null) {
			config.prngType = new PRNGRandom123("rng");
			Date date = new Date();
			config.prngSeed = date.getTime();
		} else {
			//TODO:
			config.prngType = new PRNGRandom123("rng");
			Date date = new Date();
			config.prngSeed = date.getTime();
		}
		int samplesProcessed = 0;
		int branchCount = 0;
		List<AdaptCommand> adaptCmds = new ArrayList<>();
		for (int i = 0; i < model.commandsNumber(); ++i) {
			CommandInterface cmd = model.getCommand(i);
			if (cmd instanceof AdaptCommand) {
				adaptCmds.add((AdaptCommand) cmd);
				branchCount += ((AdaptCommand) cmd).getBranchNum();
			}
		}
		final float CURRENT_STEP = 0.05f;
		CLDeviceWrapper device = currentDevices.get(0);
        mainLog.println("Using: " + device.getName());
		Sampler property = properties.get(0);
		SimulationMethod sm = property.getSimulationMethod().clone();
		// Take a copy
		Expression propNew = expr.deepCopy();
		// Combine label lists from model/property file, then expand property refs/labels in property 
		LabelList combinedLabelList = (pf == null) ? mf.getLabelList() : pf.getCombinedLabelList();
		// formulas must be expanded before replacing constants!!!
		propNew = (Expression) propNew.expandFormulas(mf.getFormulaList());
		propNew = (Expression) propNew.expandPropRefsAndLabels(pf, combinedLabelList);
		// Then get rid of any constants and simplify
		propNew = (Expression) propNew.replaceConstants(mf.getConstantValues());
		if (pf != null) {
			propNew = (Expression) propNew.replaceConstants(pf.getConstantValues());
		}
		sm.setExpression(propNew);
		int gw = 40000;
		currentContexts = new ArrayList<>();
		int GW_COUNT = 0;
		
		double current_time = 0.0;
		try {
			int m = 0;
			if (branchCount > 5) {
				m = 3;
			} else {
				m = (int) Math.floor((float) branchCount / 2) + 1;
			}
            Random rnd = new Random();
            rnd.setSeed(System.currentTimeMillis());
			float[] currentBest = new float[branchCount];
			for (int i = 0; i < branchCount; ++i) {
				currentBest[i] = rnd.nextFloat();
			}
            
            //currentBest[0] = 0.8263691f;
            //currentBest[1] = 0.8043951f;
            //currentBest[2] = 0.6847447f;
            //currentBest[3] = 0.96372974f;
            //currentBest[4] = 0.6039024f;
            //currentBest[5] = 0.8015622f;
			int max = (int) Math.pow(2, m);
			//int max = (int) Math.pow(2, branchCount);
			for (int i = 0; i < max; ++i) {
				RuntimeContext currentContext = new RuntimeContext(device, mainLog);
				currentContext.createKernel(model, properties, config);
				currentContexts.add(currentContext);
			}
			//Random rnd = new Random();
			//rnd.setSeed(System.currentTimeMillis());
			int iterations = 0;
			double simMax = 0.0;
			int posMax = -1;
			List<Pair<String, float[]>> maxes = new ArrayList<>();
			boolean gwChangeFlag = false;
			while (true) {
				int s = rnd.nextInt(m) + 1;
				//int s = rnd.nextInt(branchCoun) + 1;
				int[] changedPositions = new int[s];
				for (int i = 0; i < s; ++i) {
					changedPositions[i] = rnd.nextInt(branchCount);
				}
				int combinations = (int) Math.pow(2, s);
				Sampler[] samplers = new Sampler[combinations];
				float[][] current = new float[combinations][];
				for (int i = 0; i < combinations; ++i) {
					current[i] = new float[branchCount];
					for (int j = 0; j < branchCount; ++j) {
						current[i][j] = currentBest[j];
					}
					for (int j = 0; j < s; ++j) {
						if (((i >> j) & 1) == 1) {
							current[i][changedPositions[j]] += CURRENT_STEP;
						} else {
							current[i][changedPositions[j]] -= CURRENT_STEP;
						}
						if (current[i][changedPositions[j]] < 0) {
							current[i][changedPositions[j]] = 0;
						}
						if (current[i][changedPositions[j]] > 1.0) {
							current[i][changedPositions[j]] = 1.0f;
						}
					}
					samplers[i] = Sampler.createSampler(propNew, mf);
					samplers[i].setSimulationMethod(sm.clone());
				}
				mainLog.println(String.format("%d ITERATION, %d COMBINATIONS", iterations, combinations));
				for (int i = 0; i < combinations; ++i) {
					currentContexts.get(i).reset();
					currentContexts.get(i).setGW(gw);
					((CIMethod) samplers[i].getSimulationMethod()).setNumberOfSamples(gw);
					currentContexts.get(i).runSimulation(current[i], samplers[i], samplesProcessed);
				}
				posMax = -1;
				double oldMax = simMax;
				for (int i = 0; i < combinations; ++i) {
					SimulationMethod sm_ = samplers[i].getSimulationMethod();
					//TODO: temporal fix to avoid wrong width computation
					sm_.shouldStopNow(currentContexts.get(i).getSamplesProcessed(), samplers[i]);
					sm_.computeMissingParameterAfterSim();
					Double result = (Double) sm_.getResult(samplers[i]);
					if (simMax < result) {
						simMax = result;
						posMax = i;
					}
					mainLog.println(String.format("Result: %f", result.doubleValue()));
					mainLog.print("For: ");
					for (int j = 0; j < branchCount; ++j) {
						mainLog.print(current[i][j] + " ");
					}
					mainLog.println();
					mainLog.println(String.format("Sampling: %d samples in %d miliseconds.", currentContexts.get(i).getSamplesProcessed(),
							currentContexts.get(i).getTime()));
					current_time += currentContexts.get(i).getTime() / 1000.0;
					mainLog.println(String.format("Path length: min %d, max %d, avg %f", currentContexts.get(i).getMinPathLength(), currentContexts.get(i)
							.getMaxPathLength(), currentContexts.get(i).getAvgPathLength()));
					samplesProcessed += currentContexts.get(i).getSamplesProcessed();
				}
				if (posMax != -1) {
					currentBest = current[posMax];
					double width = (double) ((CIwidth) samplers[posMax].getSimulationMethod()).getMissingParameter();
					//"przedłużanie"
					float[] temp = Arrays.copyOf(currentBest, currentBest.length);
					float[] oldTemp = null;
					double newResult = simMax, oldResult = simMax;
					boolean flag = false;
					do {
						oldResult = newResult;
						oldTemp = Arrays.copyOf(temp, temp.length);
						for (int j = 0; j < s; ++j) {
							if (((posMax >> j) & 1) == 1) {
								temp[changedPositions[j]] += CURRENT_STEP;
							} else {
								temp[changedPositions[j]] -= CURRENT_STEP;
							}
							if (temp[changedPositions[j]] < 0) {
								flag = true;
								break;
							}
							if (temp[changedPositions[j]] > 1.0) {
								flag = true;
								break;
							}

						}
						if (flag == true)
							break;
						samplers[0].resetStats();
						currentContexts.get(0).reset();
						currentContexts.get(0).runSimulation(temp, samplers[0], samplesProcessed);
						SimulationMethod sm_ = samplers[0].getSimulationMethod();
						//TODO: temporal fix to avoid wrong width computation
						sm_.shouldStopNow(currentContexts.get(0).getSamplesProcessed(), samplers[0]);
						sm_.computeMissingParameterAfterSim();
						Double result = (Double) sm_.getResult(samplers[0]);
						newResult = result;
						if (newResult > oldResult)
							width = (double) ((CIwidth) sm_).getMissingParameter();
						mainLog.println(String.format("ADDITIONAL APPLICATION OF BEST, NEW PROBABILITY %f CURRENT BEST %f", newResult, simMax));
					} while (newResult > oldResult);
					if (oldResult > simMax) {
						currentBest = Arrays.copyOf(oldTemp, oldTemp.length);
						simMax = oldResult;
					}
					maxes.add(new Pair<String, float[]>(String.format("%d %d %f %d %f %f %f", iterations, gw, 
							current_time, samplesProcessed, simMax, simMax - width, simMax + width),
							Arrays.copyOf(currentBest, currentBest.length)));
					gwChangeFlag = false;
					GW_COUNT = 0;
					mainLog.println(String.format("MAX %f [%f,%f]", simMax, simMax - width, simMax + width));
					mainLog.print(String.format("c_i: "));
					for (int j = 0; j < branchCount; ++j) {
						mainLog.print(currentBest[j] + " ");
					}
					mainLog.println();
					mainLog.flush();
					//					if (Math.abs(simMax - oldMax) < width) {
					//						APMCiterations test = new APMCiterations(0.01, Math.abs(simMax - oldMax) / 2);
					//						test.computeMissingParameterBeforeSim();
					//						gw = (int) test.getMissingParameter();
					//						currentContexts.get(0).reset();
					//						currentContexts.get(0).setGW(gw);
					//						samplers[0].resetStats();
					//						currentContexts.get(0).runSimulation(currentBest, samplers[0], samplesProcessed);
					//						SimulationMethod sm_ = samplers[0].getSimulationMethod();
					//						//TODO: temporal fix to avoid wrong width computation
					//						sm_.shouldStopNow(currentContexts.get(0).getSamplesProcessed(), samplers[0]);
					//						sm_.computeMissingParameterAfterSim();
					//						double width2 = (double) ((CIwidth) sm_).getMissingParameter();
					//						Double result = (Double) sm_.getResult(samplers[0]);
					//						simMax = result;
					//						mainLog.println(String.format("new gw - width fix - %d current max %f [%f,%f]", gw, simMax, simMax - width2, simMax + width2));
					//						String tempStr = maxes.get(maxes.size() - 1).first
					//								+ String.format(" change GW to fix width: %d, current max [%f,%f]", gw, simMax, simMax - width2, simMax + width2);
					//						float[] tempFloat = maxes.get(maxes.size() - 1).second;
					//						maxes.remove(maxes.size() - 1);
					//						maxes.add(new Pair<String, float[]>(tempStr, tempFloat));
					//					}
				} else {
					mainLog.println("WARNING: iteration DOESN'T change anything, no improvement!");
					GW_COUNT++;
					if (GW_COUNT == 2) {
						gw *= 1.1;
						GW_COUNT = 0;
					}
					currentContexts.get(0).reset();
					currentContexts.get(0).setGW(gw);
					samplers[0].resetStats();
					currentContexts.get(0).runSimulation(currentBest, samplers[0], samplesProcessed);
					SimulationMethod sm_ = samplers[0].getSimulationMethod();
					//TODO: temporal fix to avoid wrong width computation
					sm_.shouldStopNow(currentContexts.get(0).getSamplesProcessed(), samplers[0]);
					sm_.computeMissingParameterAfterSim();
					double width = (double) ((CIwidth) sm_).getMissingParameter();
					Double result = (Double) sm_.getResult(samplers[0]);
					simMax = result;
					mainLog.println(String.format("new gw %d current max %f", gw, simMax));

					maxes.add(new Pair<String, float[]>(String.format("%d %d %f %d %f %f %f", iterations, gw, 
							current_time, samplesProcessed, simMax, simMax - width, simMax + width),
							Arrays.copyOf(currentBest, currentBest.length)));
                    //maxes.add(new Pair<String, float[]>(String.format("new gw %d current max %f [%f,%f]", gw, simMax, simMax - width, simMax + width), Arrays
					//		.copyOf(currentBest, currentBest.length)));
				}
				if (gw > 1800000)
					break;
				//if (current_time > 1000)
				//	break;
				++iterations;
				mainLog.println("-----------");
				mainLog.flush();
			}
			mainLog.println(String.format("e_max: %d I_max: %d e_increase: %f step: %f", 0, 0, 0.0f, CURRENT_STEP));
			for (int i = 0; i < maxes.size(); ++i) {
				//mainLog.print(String.format("%d: %s  c_i: ", i, maxes.get(i).first));
				mainLog.print(maxes.get(i).first + " ");
				for (int j = 0; j < branchCount; ++j) {
					mainLog.print(maxes.get(i).second[j] + " ");
				}
				mainLog.println();
			}
			//mainLog.println("Using " + currentContexts.size() + " OpenCL devices.");
		} finally {
			for (RuntimeContext context : currentContexts) {
				context.release();
				//context.currentDevice.getDevice().release();
			}
		}
		return samplesProcessed;
	}

	//	@Override
	//	public void simulateTest(PrismLog mainLog)
	//	{
	//		currentContexts = createContexts();
	//		mainLog.println("Using " + currentContexts.size() + " OpenCL devices.");
	//		for (RuntimeContext context : currentContexts) {
	//			mainLog.println(context);
	//		}
	//		for (RuntimeContext context : currentContexts) {
	//			context.createTestKernel();
	//		}
	//		for (RuntimeContext context : currentContexts) {
	//			context.runTestSimulation(mainLog);
	//		}
	//	}

	@Override
	public void setInitialState(State initialState)
	{
		this.initialState = initialState;
	}

	@Override
	public void setMaxPathLength(long maxPathLength)
	{
		this.maxPathLength = maxPathLength;
	}

	@Override
	public void setMainLog(PrismLog mainLog)
	{
		this.mainLog = mainLog;
	}

	@Override
	public void setPrismSettings(PrismSettings settings)
	{
		this.prismSettings = settings;
	}
}
