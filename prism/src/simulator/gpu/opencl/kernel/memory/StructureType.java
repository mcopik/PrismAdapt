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
package simulator.gpu.opencl.kernel.memory;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import prism.Preconditions;
import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.Include;

/**
 * 
 *
 */
public class StructureType implements VariableInterface, UDType
{
	private static class StructureValue implements CLValue
	{
		private StructureType type;
		private CLValue[] fieldsValue = null;

		public StructureValue(StructureType type, CLValue[] values)
		{
			this.type = type;
			fieldsValue = values;
		}

		@Override
		public boolean validateAssignmentTo(VariableInterface type)
		{
			if (type instanceof StructureType) {
				StructureType structure = (StructureType) type;
				return structure.compatibleWithStructure(this.type);
			} else {
				return false;
			}
		}

		@Override
		public Expression getSource()
		{
			StringBuilder builder = new StringBuilder("{\n");
			for (CLValue value : fieldsValue) {
				builder.append(value.getSource().toString());
				builder.append(",\n");
			}
			int len = builder.length();
			builder.delete(len - 2, len - 1);
			builder.append("}");
			return new Expression(builder.toString());
		}
	}

	private Map<String, CLVariable> fields = new LinkedHashMap<>();
	public final String typeName;

	public StructureType(String typeName)
	{
		this.typeName = typeName;
	}

	public void addVariable(CLVariable var)
	{
		fields.put(var.varName, var);
	}

	public Collection<CLVariable> getFields()
	{
		return fields.values();
	}

	public Expression getDeclaration()
	{
		StringBuilder builder = new StringBuilder();
		builder.append("typedef struct _").append(typeName).append(" ");
		builder.append(typeName).append(";");
		return new Expression(builder.toString());
	}

	public boolean compatibleWithStructure(StructureType type)
	{
		if (fields.size() != type.fields.size()) {
			return false;
		}
		Iterator<CLVariable> thisIt = fields.values().iterator();
		Iterator<CLVariable> otherIt = type.fields.values().iterator();
		while (thisIt.hasNext()) {
			if (!thisIt.next().equals(otherIt.next())) {
				return false;
			}
		}
		return true;
	}

	public boolean containsField(String fieldName)
	{
		return fields.containsKey(fieldName);
	}

	public CLValue initializeStdStructure(Number[] values)
	{
		CLValue[] init = new CLValue[values.length];
		for (int i = 0; i < values.length; ++i) {
			init[i] = StdVariableType.initialize(values[i]);
		}
		return new StructureValue(this, init);
	}

	@Override
	public String getType()
	{
		return typeName;
	}

	public Expression getDefinition()
	{
		StringBuilder builder = new StringBuilder();
		builder.append("typedef struct _").append(typeName).append("{\n");
		for (CLVariable var : fields.values()) {
			builder.append(var.getDeclaration());
			builder.append("\n");
		}
		builder.append("} ").append(typeName).append(";\n");
		return new Expression(builder.toString());
	}

	@Override
	public List<Include> getIncludes()
	{
		return null;
	}

	@Override
	public boolean isStructure()
	{
		return true;
	}

	@Override
	public boolean isArray()
	{
		return false;
	}

	@Override
	public CLVariable accessElement(CLVariable var, Expression index)
	{
		return null;
	}

	@Override
	public CLVariable accessField(String varName, String fieldName)
	{
		Preconditions.checkCondition(containsField(fieldName), String.format("StructureType %s does not contain field %s", typeName, fieldName));
		CLVariable var = fields.get(fieldName);
		return new CLVariable(var.varType, varName + "." + fieldName);
	}

	@Override
	public String declareVar(String varName)
	{
		return String.format("%s %s", typeName, varName);
	}
}