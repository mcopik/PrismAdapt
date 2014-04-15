/**
 * 
 */
package simulator.gpu.opencl.kernel;

import java.util.ArrayList;
import java.util.List;

import org.bridj.Pointer;

import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.ExpressionList;
import simulator.gpu.opencl.kernel.expression.Include;
import simulator.gpu.opencl.kernel.expression.KernelComponent;
import simulator.gpu.opencl.kernel.expression.Method;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.CLVariable.Location;
import simulator.gpu.opencl.kernel.memory.PointerType;
import simulator.gpu.opencl.kernel.memory.RValue;
import simulator.gpu.opencl.kernel.memory.StdVariableType;
import simulator.gpu.opencl.kernel.memory.StdVariableType.StdType;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;

/**
 * @author mcopik
 *
 */
public class PRNGRandom123 extends PRNGType
{
	private static final List<Include> INCLUDES = new ArrayList<>();
	static {
		INCLUDES.add(new Include("Random123/threefry.h", true));
	}
	private static final List<CLVariable> ADDITIONAL_ARGS = new ArrayList<>();
	private static final List<KernelComponent> ADDITIONAL_DEFINITIONS = new ArrayList<>();
	public static final String RNG_MAX = "0xFFFFFFFF";
	static {
		try {
			ADDITIONAL_DEFINITIONS.add(new Expression("#define R123_0x1p_31f (1.f/(1024.f*1024.f*1024.f*2.f))"));
			ADDITIONAL_DEFINITIONS.add(new Expression("#define R123_0x1p_24f (128.f*R123_0x1p_31f)"));
			Method convert = new Method("u01fixedpt_closed_open_32_24", new StdVariableType(StdType.FLOAT));
			convert.addArg(new CLVariable(new StdVariableType(StdType.UINT32), "i"));
			convert.addReturn(new Expression("(i>>8)*R123_0x1p_24f"));
			ADDITIONAL_DEFINITIONS.add(convert);
		} catch (KernelException e) {
			throw new RuntimeException("Fatal error during static class initialization!");
		}
	}
	private final static boolean seedUseGlobalMemory = true;
	static {
		try {
			CLVariable seed;
			if (seedUseGlobalMemory) {
				seed = new CLVariable(new PointerType(new StdVariableType(StdType.UINT32)), "state");
				seed.setMemoryLocation(Location.GLOBAL);
			} else {
				//TODO: javacl gives invalidargsize?
				seed = new CLVariable(new StdVariableType(StdType.INT32, 3), "state");
			}
			ADDITIONAL_ARGS.add(seed);
		} catch (KernelException e) {
			throw new RuntimeException("Fatal error during static class initialization!");
		}
	}
	private int[] seed = null;
	private CLBuffer<Integer> initBuffer = null;

	public PRNGRandom123(String varName)
	{
		super(varName, INCLUDES, ADDITIONAL_ARGS);
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.PRNGType#initializeGenerator()
	 */
	@Override
	public KernelComponent initializeGenerator()
	{
		ExpressionList list = new ExpressionList();
		CLVariable state = ADDITIONAL_ARGS.get(0);
		String counter;
		if (!seedUseGlobalMemory) {
			counter = String.format("threefry2x32_ctr_t ctr = {{0,sampleNumber+globalID+%s.s0}};", state.varName);
		} else {
			counter = String.format("threefry2x32_ctr_t ctr = {{0,sampleNumber+globalID+%s[0]}};", state.varName);
		}
		list.addExpression(new Expression(counter));
		String key;
		if (!seedUseGlobalMemory) {
			key = String.format("threefry2x32_key_t key = {{%s.s1,%s.s2}};", state.varName, state.varName);
		} else {
			key = String.format("threefry2x32_key_t key = {{%s[1],%s[2]}};", state.varName, state.varName);
		}
		list.addExpression(new Expression(key));
		String rand = String.format("threefry2x32_ctr_t rand;", state.varName, state.varName);
		list.addExpression(new Expression(rand));
		return list;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.PRNGType#numbersPerRandomize()
	 */
	@Override
	public int numbersPerRandomize()
	{
		return 2;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.PRNGType#randomize()
	 */
	@Override
	public KernelComponent randomize() throws KernelException
	{
		ExpressionList list = new ExpressionList();
		list.addExpression(new Expression("rand = threefry2x32(ctr, key);"));
		list.addExpression(new Expression("ctr.v[0]++;"));
		return list;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.PRNGType#assignRandomInt(int, simulator.gpu.opencl.kernel.memory.CLVariable, int)
	 */
	@Override
	public CLValue getRandomInt(Expression randNumber, Expression max)
	{
		return new RValue(new Expression(String.format("rand.v[%s]*%s/%s",
		//counter
				randNumber.getSource(),
				//current max for int
				max.getSource(),
				//random max
				RNG_MAX)));
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.PRNGType#assignRandomFloat(int, simulator.gpu.opencl.kernel.memory.CLVariable, simulator.gpu.opencl.kernel.memory.CLVariable)
	 */
	@Override
	public CLValue getRandomFloat(Expression randNumber, Expression max)
	{
		return new RValue(new Expression(String.format("u01fixedpt_closed_open_32_24(rand.v[%s])*%s",
		//counter
				randNumber.getSource(),
				//random max
				max.getSource())));
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.PRNGType#assignRandomFloat(int, simulator.gpu.opencl.kernel.memory.CLVariable)
	 */
	@Override
	public CLValue getRandomUnifFloat(Expression randNumber)
	{
		return new RValue(new Expression(String.format("u01fixedpt_closed_open_32_24(rand.v[%s])",
		//counter
				randNumber.getSource())));
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.PRNGType#setKernelArg(com.nativelibs4java.opencl.CLKernel, int, long, int, int)
	 */
	@Override
	public void setKernelArg(CLKernel kernel, int argNumber, int sampleOffset, int globalWorkSize, int localWorkSize)
	{
		if (seed == null) {
			seed = new int[3];
			seed[0] = (int) Math.floor(random.randomUnifDouble() * Integer.MAX_VALUE);
			seed[1] = (int) Math.floor(random.randomUnifDouble() * Integer.MAX_VALUE);
			seed[2] = (int) Math.floor(random.randomUnifDouble() * Integer.MAX_VALUE);
			if (seedUseGlobalMemory) {
				Pointer<Integer> ptr = Pointer.allocateInts(3);
				ptr.setInts(seed);
				CLContext context = kernel.getProgram().getContext();
				initBuffer = context.createIntBuffer(CLMem.Usage.InputOutput, ptr, true);
				kernel.setArg(0, initBuffer);

			} else {
				kernel.setArg(0, seed);
			}
		} else {
			Pointer<Integer> ptr = Pointer.allocateInts(3);
			ptr.setInts(seed);
			CLContext context = kernel.getProgram().getContext();
			initBuffer = context.createIntBuffer(CLMem.Usage.InputOutput, ptr, true);
			kernel.setArg(0, initBuffer);
		}
	}

	public List<KernelComponent> getAdditionalDefinitions()
	{
		return ADDITIONAL_DEFINITIONS;
	}
}
